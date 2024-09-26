# Steps to reproduce

```bash
git clone https://github.com/leifdenby/uclales
cd uclales
git checkout advective-trajectories
mamba env create -f .github/ci_deps/Linux.yml -n uclales
conda activate uclales
cmake -DMPI=TRUE .
make -j4
mkdir run
cd run
# create NAMELIST, see below
mpiexec -np 4 ../
tar zcvf uclales.advtraj.tesdata.tar.gz NAMELIST rico.????????.nc
```


NAMELIST:
```
&model
 nxp = 36
 nyp = 36
 nzp = 70
 nxpart = .true.
 deltax  =100.
 deltay  =100.
 deltaz  = 40.
 dzmax   = 10.
 dzrat   = 1.0
 dtlong  = 4.
 timmax =  240.
 runtype= 'INITIAL'
 filprf = 'rico'
 hfilin = 'rico.rst'
 savg_intvl=120.
 ssam_intvl=  30.
 frqhis = 60.
 frqanl = 120.
 level  = 3
 CCN    = 70.e6
 corflg = .true.
 cntlat = 18.
 th00   = 299.8
 sst    = 299.8
 isfctyp=3
 iradtyp=1
 zrough = 0.001229
 dthcon = 0.001094
 drtcon = 0.001133
 ps =    1015.4,  740., 3260., 4000.,
 ts =     297.9, 297.9, 312.6644, 317.0,
 rts=      16.0,  13.8,   2.4,   1.8,
 us =      -9.9, -8.42,  -3.38, -1.9,
 vs =      -3.8,  -3.8,  -3.8,  -3.8,
 vmean = -4.
 umean = -5.
 case_name='rico'
 lsvarflg=.false.
 div=0.00000000

 ! turn on optional features
 ladvtrc = .true.  ! advected position tracers
/
```
