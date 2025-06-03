# mpiexec -n 8 /appalachia/d4/shjzhang/test_stanley_src/Athena_Radiation_HydroFreezeHybrid_Stanley/bin/athena -i athinput.disk_sph_rad
mpiexec -n 8 /appalachia/d4/shjzhang/test_stanley_src/Athena_Radiation_HydroFreezeHybrid_Stanley/bin/athena -r disk.final.rst radiation/cfl_rad=3e-4
# mpiexec -n 4 /Users/shjzhang/github/Athena0630/Athena_Radiation_HydroFreezeHybrid_Stanley/bin/athena -r disk.final.rst
