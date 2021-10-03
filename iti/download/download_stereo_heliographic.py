import datetime
import os

import astropy.units as u
from sunpy.net import Fido
from sunpy.net import attrs as a

wavelengths = [(171, 171), (195, 193,), (284, 211,), (304, 304,)]

t_start = datetime.datetime(2010, 12, 12)

while t_start < datetime.datetime(2011, 1, 1):
    try:
        for stereo_wl, sdo_wl in wavelengths:
            stereo_a = Fido.search((a.Instrument("EUVI") & a.Source('STEREO_A') &
                                    a.Time(t_start, t_start + datetime.timedelta(hours=1)) &
                                    a.Wavelength(stereo_wl * u.AA)))
            if stereo_a.file_num == 0:
                raise Exception('No data found %s' % t_start)
            stereo_a = stereo_a[:, 0]
            ref_time = stereo_a[0, 0]['Start Time']
            stereo_b = Fido.search((a.Instrument("EUVI") & a.Source('STEREO_B') &
                                    a.Time(t_start, t_start + datetime.timedelta(hours=1), near=ref_time) &
                                    a.Wavelength(stereo_wl * u.AA)))[:, 0]
            sdo = Fido.search((a.Instrument.aia &
                               a.Time(t_start, t_start + datetime.timedelta(hours=1), near=ref_time) &
                               a.Wavelength(sdo_wl * u.AA)))[:, 0]
            base_path = '/localdata/USER/rja/stereo_heliographic/%d' % stereo_wl
            file = Fido.fetch(stereo_a, path=base_path)[0]
            os.rename(file, base_path + '/%s_A.fits' % t_start.isoformat('T'))
            file = Fido.fetch(stereo_b, path=base_path)[0]
            os.rename(file, base_path + '/%s_B.fits' % t_start.isoformat('T'))
            file = Fido.fetch(sdo, path=base_path)[0]
            os.rename(file, base_path + '/%s_sdo.fits' % t_start.isoformat('T'))

        # download HMI magnetogram
        hmi_res = Fido.search((a.Instrument.hmi &
                               a.Time(t_start, t_start + datetime.timedelta(hours=1), near=ref_time) &
                               a.Physobs('LOS_magnetic_field')))[:, 1]  # workaround synoptic maps
        base_path = '/localdata/USER/rja/stereo_heliographic/mag'
        file = Fido.fetch(hmi_res, path=base_path)[0]
        os.rename(file, base_path + '/%s_sdo.fits' % t_start.isoformat('T'))
        # increase for next iteration
        t_start += datetime.timedelta(hours=24)
    except Exception as ex:
        print(ex)
        t_start += datetime.timedelta(hours=24)

# shutil.unpack_archive('/gss/r.jarolim/data/stereo_heliographic_prep.zip', '/gss/r.jarolim/data/stereo_heliographic_prep')
# [shutil.move(f, f.replace('stereo_heliographic', 'stereo_heliographic_prep')) for f in glob.glob('/localdata/USER/rja/stereo_heliographic/**/*sdo.fits')]
# shutil.make_archive('/localdata/USER/rja/stereo_heliographic_prep', 'zip', '/localdata/USER/rja/stereo_heliographic_prep')


# t_start = datetime.datetime(2010, 12, 3)
# while t_start < datetime.datetime(2010, 12, 12):
#     hmi_res = Fido.search((a.Instrument.hmi &
#            a.Time(t_start, t_start + datetime.timedelta(minutes=5)) &
#            a.Physobs('LOS_magnetic_field')))[:, 1]
#     base_path = '/gss/r.jarolim/data/stereo_heliographic_prep/mag'
#     file = Fido.fetch(hmi_res, path=base_path)[0]
#     os.rename(file, base_path + '/%s_sdo.fits' % t_start.isoformat('T'))
#     # increase for next iteration
#     t_start += datetime.timedelta(hours=24)
