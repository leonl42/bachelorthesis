

rm -r ./data/adam_wbn/norm/3.0_1/states
rm -r ./data/adam_wbn/gcstdu/3.0_1/states
rm -r ./data/adam_wbn/norm/3.0_1/mg_spacing_0
rm -r ./data/adam_wbn/gcstdu/3.0_1/mg_spacing_0
mkdir ./data/adam_wbn/norm/3.0_1/states
mkdir ./data/adam_wbn/gcstdu/3.0_1/states

rsync hpc:/share/users/student/l/llemke/bachelorthesis/data/adam_wbn/norm/3.0_1/states/0.pkl ./data/adam_wbn/norm/3.0_1/states/0.pkl --info=progress2
rsync hpc:/share/users/student/l/llemke/bachelorthesis/data/adam_wbn/gcstdu/3.0_1/states/0.pkl ./data/adam_wbn/gcstdu/3.0_1/states/0.pkl --info=progress2

rsync -a hpc:/share/users/student/l/llemke/bachelorthesis/data/adam_wbn/norm/3.0_1/mg_spacing_0 ./data/adam_wbn/norm/3.0_1 --info=progress2
rsync -a hpc:/share/users/student/l/llemke/bachelorthesis/data/adam_wbn/gcstdu/3.0_1/mg_spacing_0 ./data/adam_wbn/gcstdu/3.0_1 --info=progress2