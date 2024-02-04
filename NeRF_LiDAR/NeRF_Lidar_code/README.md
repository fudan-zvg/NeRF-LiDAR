Command:
## Drop rays
<!-- For placement:
```
python src/drop_simulation_rays.py --expname xxxx --start xxxx --simulation_path DATA/allsimulation/xxxx --dvgo --place_car
```
No placement: -->
```
python src/drop_simulation_rays.py --expname xxxx --start xxxx --simulation_path DATA/allsimulation/xxxx --dvgo
```
## Training raydrop

```
From scratch:
python src/transfer_lidar_data.py --expname xxxx --dvgo --moving_mask --ray_drop xxxx --vgg --vgg_weights 0.2 --mix_train --batch_size 4 # train a model

Already load data:
python src/transfer_lidar_data.py --expname xxxx --dvgo --moving_mask --ray_drop xxxx --vgg --vgg_weights 0.2 --mix_train --batch_size 4 --load_data # skip loading data

```
