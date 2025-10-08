from . import geom_provider as gp

dist = gp.get_district_data()
upz = gp.get_upazila_data()

UPAZILA_NAMES = {dist : [] for dist in dist.DISTRICT_N.unique()}
for _, row in upz.iterrows():
    UPAZILA_NAMES[row.DISTRICT_N].append(row.UPAZILA_NA)


def get_districts_list():
    """Get list of available districts"""
    return sorted(dist.DISTRICT_N.unique().tolist())

def get_upazilas_by_district(district_name: str):
    """Get list of upazilas for a specific district"""
    filtered_upz = upz[upz['DISTRICT_N'] == district_name]
    return sorted(filtered_upz['UPAZILA_NA'].unique().tolist())