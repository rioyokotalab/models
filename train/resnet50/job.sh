python resnet50_trainer.py --train_data /mnt/nas101/hiroki11x/ilsvrc12_train_lmdb --gpus 0 \
 --batch_size 32 \
 --epoch_size 50000 \
 --num_epoch 1000000 \
 2>&1 | tee

