document.addEventListener('DOMContentLoaded', function() {
    // Navigation elements
    const navbar = document.getElementById('navbar');
    const navToggle = document.getElementById('navToggle');
    const navMenu = document.getElementById('navMenu');
    const navLinks = document.querySelectorAll('.nav-link');

    // Get DOM elements
    const selectAllBtn = document.getElementById('selectAllBtn');
    const runAnalysisBtn = document.getElementById('runAnalysisBtn');
    const statusMessage = document.getElementById('statusMessage');
    const districtSelect = document.getElementById('district');
    const upazilaSelect = document.getElementById('upazila');
    const checkboxes = document.querySelectorAll('input[name="analysis[]"]');
    const btnText = runAnalysisBtn.querySelector('.btn-text');
    const btnLoading = runAnalysisBtn.querySelector('.btn-loading');

    // State variables
    let isAnalyzing = false;
    let allSelected = false;
    let websocket = null;
    let sessionId = generateSessionId();
    let analysisResults = [];
    let buttonMode = 'analysis'; // 'analysis' or 'results'

    // Initialize
    updateSelectAllButton();
    updateRunAnalysisButton();
    
    // Load districts on page load
    loadDistricts();

    // Navigation functionality
    initializeNavigation();

    // Event listeners
    selectAllBtn.addEventListener('click', toggleSelectAll);
    runAnalysisBtn.addEventListener('click', runAnalysis);
    districtSelect.addEventListener('change', handleDistrictChange);
    upazilaSelect.addEventListener('change', updateRunAnalysisButton);
    
    // Add event listeners to checkboxes
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            updateSelectAllButton();
            updateRunAnalysisButton();
        });
    });

    // Navigation Functions
    function initializeNavigation() {
        // Handle scroll events for navbar transformation
        window.addEventListener('scroll', handleNavbarScroll);
        
        // Handle mobile menu toggle
        if (navToggle) {
            navToggle.addEventListener('click', toggleMobileMenu);
        }
        
        // Close mobile menu when clicking on links
        navLinks.forEach(link => {
            link.addEventListener('click', closeMobileMenu);
        });
        
        // Close mobile menu when clicking outside
        document.addEventListener('click', function(event) {
            if (!navbar.contains(event.target)) {
                closeMobileMenu();
            }
        });

        // Handle smooth scrolling for internal links
        navLinks.forEach(link => {
            if (link.getAttribute('href').startsWith('#')) {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const targetId = this.getAttribute('href').substring(1);
                    const targetElement = document.getElementById(targetId);
                    if (targetElement) {
                        targetElement.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                });
            }
        });
    }

    function handleNavbarScroll() {
        const scrollY = window.scrollY;
        if (scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    }

    function toggleMobileMenu() {
        navToggle.classList.toggle('active');
        navMenu.classList.toggle('active');
    }

    function closeMobileMenu() {
        navToggle.classList.remove('active');
        navMenu.classList.remove('active');
    }

    function reconnectWebSocket(protocol) {
        console.log("Reconnecting to WebSocket at protocol:", protocol);
        const wsUrl = `${protocol}//${window.location.host}/ws-reconnect/${sessionId}`;
        
        websocket = new WebSocket(wsUrl);

        websocket.onopen = function(event) {
            showNotification('Reconnected to server', 'success');
            console.log('WebSocket reconnected');
        }

        websocket.onmessage = function(event) {
            const message = JSON.parse(event.data);
            handleWebSocketMessage(message);
        };

        websocket.onclose = function(event) {
            console.log('WebSocket disconnected');
            if (event && event.code && event.code == 1012){
                showNotification('Analysis failed unexpectedly');
                resetForm();
                isAnalyzing = false;
                return;
            }
            if (isAnalyzing) {
                // Attempt to reconnect if analysis is still running
                setTimeout(reconnectWebSocket(protocol), 10000);

            }
        };

        websocket.onerror = function(error) {
            console.error('WebSocket error:', error);
            if (error && error.code == 1012) {
                showNotification('Analysis failed unexpectedly');
                resetForm();
            }
        }

    }


    // WebSocket functions
    function connectWebSocket() {
        const protocol = (window.location.protocol === 'https:') ? 'wss:' : 'ws:';
        console.log("Connecting to WebSocket at protocol:", protocol);
        const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;
        
        websocket = new WebSocket(wsUrl);
        
        websocket.onopen = function(event) {
            console.log('WebSocket connected');
        };
        
        websocket.onmessage = function(event) {
            const message = JSON.parse(event.data);
            handleWebSocketMessage(message);
        };
        
        websocket.onclose = function(event) {
            showNotification('Connection Lost', 'warning');
            console.log('WebSocket disconnected');
            if (isAnalyzing) {
                // Attempt to reconnect if analysis is still running
                setTimeout(reconnectWebSocket(protocol), 10000);
                return;
            }
        };
        
        websocket.onerror = function(error) {
            console.error('WebSocket error:', error);
        };
    }

    let currentProgressItem;
    
    function handleWebSocketMessage(message) {
        console.log('Received message:', message);
        
        switch(message.type) {
            case 'analysis_start':
                currentProgressItem = showAnalysisProgress(message.message, 'info');
                break;
                
            case 'progress_update':
                updateProgressBar(message.current, message.total, message.message);
                break;
                
            case 'analysis_complete':
                markProgressAsCompleted(message.message, currentProgressItem);
                break;
                
            case 'analysis_error':
                showAnalysisProgress(message.message, 'error');
                break;
                
            case 'all_analyses_complete':
                handleAnalysisComplete(message);
                break;
                
            case 'error':
                showNotification(message.message, 'error');
                resetForm();
                break;
        }
    }
    
    function handleAnalysisComplete(message) {
        isAnalyzing = false;
        analysisResults = message.completed_analyses;
        buttonMode = 'results';
        
        // Close WebSocket connection since analysis is complete
        if (websocket) {
            websocket.close();
            websocket = null;
            console.log('WebSocket closed after analysis completion');
        }
        
        // Update UI
        btnText.style.display = 'flex';
        btnLoading.style.display = 'none';
        
        // Change button to "View Results" and remove the click event listener
        btnText.innerHTML = '<i class="fas fa-chart-bar"></i> View Results';
        runAnalysisBtn.removeEventListener('click', runAnalysis);
        runAnalysisBtn.onclick = null; // Clear any existing onclick
        runAnalysisBtn.addEventListener('click', showResults);
        runAnalysisBtn.disabled = false;
        runAnalysisBtn.classList.add('results-ready');
        
        // Update status message
        statusMessage.innerHTML = `
            <i class="fas fa-check-circle"></i>
            Analysis complete! ${message.completed_analyses.length} results ready to view.
        `;
        statusMessage.classList.add('success');
        statusMessage.style.display = 'flex';
        
        // Re-enable form elements
        locationInput.disabled = false;
        selectAllBtn.disabled = false;
        checkboxes.forEach(cb => cb.disabled = false);
        
        showNotification(message.message, 'success');
    }

    // Utility functions
    function generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    function showAnalysisProgress(message, type = 'info') {
        const progressContainer = document.getElementById('progressContainer') || createProgressContainer();
        
        const progressItem = document.createElement('div');
        progressItem.className = `progress-item progress-${type}`;
        progressItem.innerHTML = `
            <i class="fas fa-${getProgressIcon(type)}"></i>
            <span>${message}</span>
            <span class="timestamp">${new Date().toLocaleTimeString()}</span>
        `;
        
        progressContainer.appendChild(progressItem);
        progressContainer.scrollTop = progressContainer.scrollHeight;
        return progressItem;
    }

    function markProgressAsCompleted(message, progressItem){
        progressItem.innerHTML = `
            <i class="fas fa-${getProgressIcon("success")}"></i>
            <span>${message}</span>
            <span class="timestamp">${new Date().toLocaleTimeString()}</span>
        `;
    }
    
    function createProgressContainer() {
        const container = document.createElement('div');
        container.id = 'progressContainer';
        container.className = 'progress-container';
        container.innerHTML = '<h4><i class="fas fa-list"></i> Analysis Progress</h4>';
        
        statusMessage.insertAdjacentElement('afterend', container);
        return container;
    }
    
    function updateProgressBar(current, total, message) {
        let progressBar = document.getElementById('analysisProgressBar');
        
        if (!progressBar) {
            progressBar = document.createElement('div');
            progressBar.id = 'analysisProgressBar';
            progressBar.className = 'analysis-progress-bar';
            progressBar.innerHTML = `
                <div class="progress-bar-container">
                    <div class="progress-bar-fill"></div>
                </div>
                <div class="progress-text"></div>
            `;
            statusMessage.insertAdjacentElement('afterend', progressBar);
        }
        
        const percentage = (current / total) * 100;
        const fill = progressBar.querySelector('.progress-bar-fill');
        const text = progressBar.querySelector('.progress-text');
        
        fill.style.width = `${percentage}%`;
        text.textContent = `${message} (${current}/${total})`;
    }
    
    function getProgressIcon(type) {
        switch(type) {
            case 'success': return 'check';
            case 'error': return 'times';
            case 'info': return 'spinner fa-spin';
            default: return 'info-circle';
        }
    }

    // Modified functions
    function toggleSelectAll() {
        if (isAnalyzing) return;
        
        allSelected = !allSelected;
        checkboxes.forEach(checkbox => {
            checkbox.checked = allSelected;
        });
        
        updateSelectAllButton();
        updateRunAnalysisButton();
        
        // Add visual feedback
        selectAllBtn.style.transform = 'scale(0.95)';
        setTimeout(() => {
            selectAllBtn.style.transform = 'scale(1)';
        }, 150);
    }

    function updateSelectAllButton() {
        const checkedCount = Array.from(checkboxes).filter(cb => cb.checked).length;
        const totalCount = checkboxes.length;
        
        allSelected = checkedCount === totalCount;
        
        if (allSelected) {
            selectAllBtn.innerHTML = '<i class="fas fa-check-double"></i> Deselect All';
            selectAllBtn.style.background = 'var(--primary-color)';
            selectAllBtn.style.color = 'white';
            selectAllBtn.style.borderColor = 'var(--primary-color)';
        } else {
            selectAllBtn.innerHTML = '<i class="fas fa-check-double"></i> Select All';
            selectAllBtn.style.background = 'var(--bg-accent)';
            selectAllBtn.style.color = 'var(--text-secondary)';
            selectAllBtn.style.borderColor = 'var(--border-color)';
        }
    }

    function updateRunAnalysisButton() {
        if (isAnalyzing) return;
        
        const hasDistrict = districtSelect.value.trim().length > 0;
        const hasSelectedAnalysis = Array.from(checkboxes).some(cb => cb.checked);
        
        const canRun = hasDistrict && hasSelectedAnalysis;
        
        runAnalysisBtn.disabled = !canRun;
        
        if (!hasDistrict && !hasSelectedAnalysis) {
            runAnalysisBtn.title = 'Please select a district and at least one analysis option';
        } else if (!hasDistrict) {
            runAnalysisBtn.title = 'Please select a district';
        } else if (!hasSelectedAnalysis) {
            runAnalysisBtn.title = 'Please select at least one analysis option';
        } else {
            runAnalysisBtn.title = '';
        }
    }

    function runAnalysis() {
        if (isAnalyzing || buttonMode === 'results') return;
        
        const district = districtSelect.value.trim();
        const upazila = upazilaSelect.value.trim() || null;
        const selectedAnalysis = Array.from(checkboxes)
            .filter(cb => cb.checked)
            .map(cb => cb.value);
        
        if (!district || selectedAnalysis.length === 0) {
            showNotification('Please select a district and at least one analysis option', 'error');
            return;
        }
        
        startAnalysis(selectedAnalysis, district, upazila);
    }
    
    async function startAnalysis(selectedAnalysis, district, upazila) {
        try {
            isAnalyzing = true;
            
            // Connect to WebSocket
            connectWebSocket();
            
            // Update UI
            btnText.style.display = 'none';
            btnLoading.style.display = 'flex';
            runAnalysisBtn.disabled = true;
            runAnalysisBtn.classList.remove('results-ready');
            
            // Show status message
            statusMessage.innerHTML = `
                <i class="fas fa-clock"></i>
                The analysis may take a while. Hold tight...
            `;
            statusMessage.classList.remove('success');
            statusMessage.style.display = 'flex';
            
            // Disable form elements
            districtSelect.disabled = true;
            upazilaSelect.disabled = true;
            selectAllBtn.disabled = true;
            checkboxes.forEach(cb => cb.disabled = true);
            
            // Clear any previous progress
            const existingProgress = document.getElementById('progressContainer');
            if (existingProgress) existingProgress.remove();
            
            const existingProgressBar = document.getElementById('analysisProgressBar');
            if (existingProgressBar) existingProgressBar.remove();
            
            // Send request to backend
            body = {
                    district: district,
                    analyses: selectedAnalysis,
                    session_id: sessionId
                }
            if (upazila) {
                body.upazila = upazila;
            }
            const response = await fetch('/run-analysis', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: body ? JSON.stringify(body) : null,
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            console.log('Analysis started:', result);
            
        } catch (error) {
            console.error('Error starting analysis:', error);
            showNotification('Error starting analysis: ' + error.message, 'error');
            resetForm();
        }
    }
    
    function showResults() {
        console.log('Redirecting to results viewer with session ID:', sessionId);
        // Redirect to results viewer with session ID
        window.location.href = `/results-viewer?session=${sessionId}`;
    }
    
    function resetForm() {
        isAnalyzing = false;
        buttonMode = 'analysis';
        
        // Close WebSocket if connected
        if (websocket) {
            websocket.close();
            websocket = null;
            console.log('WebSocket closed during form reset');
        }
        
        // Reset button state
        btnText.innerHTML = '<i class="fas fa-play"></i> Run Analysis';
        btnText.style.display = 'flex';
        btnLoading.style.display = 'none';
        
        // Remove any existing event listeners and reset to original function
        runAnalysisBtn.removeEventListener('click', showResults);
        runAnalysisBtn.onclick = null;
        runAnalysisBtn.addEventListener('click', runAnalysis);
        runAnalysisBtn.disabled = false;
        runAnalysisBtn.classList.remove('results-ready');
        
        // Hide status message
        statusMessage.style.display = 'none';
        statusMessage.classList.remove('success');
        
        // Re-enable form elements
        districtSelect.disabled = false;
        upazilaSelect.disabled = districtSelect.value ? false : true;
        selectAllBtn.disabled = false;
        checkboxes.forEach(cb => cb.disabled = false);
        
        // Clean up progress elements
        const progressContainer = document.getElementById('progressContainer');
        if (progressContainer) progressContainer.remove();
        
        const progressBar = document.getElementById('analysisProgressBar');
        if (progressBar) progressBar.remove();
        
        updateRunAnalysisButton();
    }
    
    function showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
        `;
        
        // Style notification
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            background: getNotificationColor(type),
            color: 'white',
            padding: '1rem 1.5rem',
            borderRadius: '0.5rem',
            boxShadow: 'var(--shadow-lg)',
            zIndex: '1000',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontSize: '0.875rem',
            fontWeight: '500',
            maxWidth: '300px',
            transform: 'translateX(100%)',
            transition: 'transform 0.3s ease'
        });
        
        // Add to DOM
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 10);
        
        // Remove after delay
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 4000);
    }
    
    function getNotificationIcon(type) {
        switch (type) {
            case 'success': return 'check-circle';
            case 'error': return 'exclamation-circle';
            case 'warning': return 'exclamation-triangle';
            default: return 'info-circle';
        }
    }
    
    function getNotificationColor(type) {
        switch (type) {
            case 'success': return 'var(--success-color)';
            case 'error': return 'var(--danger-color)';
            case 'warning': return 'var(--warning-color)';
            default: return 'var(--primary-color)';
        }
    }
    
    // // Add some nice interactions
    // locationInput.addEventListener('focus', function() {
    //     this.parentElement.style.transform = 'translateY(-2px)';
    // });
    
    // locationInput.addEventListener('blur', function() {
    //     this.parentElement.style.transform = 'translateY(0)';
    // });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to run analysis or show results
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            if (!runAnalysisBtn.disabled && !isAnalyzing) {
                if (buttonMode === 'results') {
                    showResults();
                } else {
                    runAnalysis();
                }
            }
        }
        
        // Ctrl/Cmd + A to select all (when not focused on inputs)
        if ((e.ctrlKey || e.metaKey) && e.key === 'a' && 
            document.activeElement !== districtSelect && 
            document.activeElement !== upazilaSelect) {
            e.preventDefault();
            if (!isAnalyzing) {
                toggleSelectAll();
            }
        }
    });
    
    // Functions for district and upazila handling
    async function loadDistricts() {
        try {
            const response = await fetch('/api/districts');
            const data = await response.json();
            
            // Clear existing options except the first one
            districtSelect.innerHTML = '<option value="">Select a district</option>';
            
            // Add districts to dropdown
            data.districts.forEach(district => {
                const option = document.createElement('option');
                option.value = district;
                option.textContent = district;
                districtSelect.appendChild(option);
            });
            
            // Set default to Narayanganj if available
            if (data.districts.includes('Narayanganj')) {
                districtSelect.value = 'Narayanganj';
                loadUpazilas('Narayanganj');
            }
            
            updateRunAnalysisButton();
        } catch (error) {
            console.error('Error loading districts:', error);
            showNotification('Failed to load districts', 'error');
        }
    }
    
    async function loadUpazilas(districtName) {
        try {
            upazilaSelect.disabled = true;
            upazilaSelect.innerHTML = '<option value="">Loading...</option>';
            
            const response = await fetch(`/api/upazilas/${encodeURIComponent(districtName)}`);
            const data = await response.json();
            
            // Clear existing options
            upazilaSelect.innerHTML = '<option value="">Select a upazila (optional)</option>';
            
            // Add upazilas to dropdown
            data.upazilas.forEach(upazila => {
                const option = document.createElement('option');
                option.value = upazila;
                option.textContent = upazila;
                upazilaSelect.appendChild(option);
            });
            
            upazilaSelect.disabled = false;
            updateRunAnalysisButton();
        } catch (error) {
            console.error('Error loading upazilas:', error);
            upazilaSelect.innerHTML = '<option value="">Error loading upazilas</option>';
            upazilaSelect.disabled = false;
            showNotification('Failed to load upazilas', 'error');
        }
    }
    
    function handleDistrictChange() {
        const selectedDistrict = districtSelect.value;
        
        if (selectedDistrict) {
            loadUpazilas(selectedDistrict);
        } else {
            upazilaSelect.innerHTML = '<option value="">Select a upazila (optional)</option>';
            upazilaSelect.disabled = true;
        }
        
        updateRunAnalysisButton();
    }
});