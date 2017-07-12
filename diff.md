diff --git a/train/resnet50/resnet.py b/train/resnet50/resnet.py
index 8bb7e5c..1b376f8 100644
--- a/train/resnet50/resnet.py
+++ b/train/resnet50/resnet.py
@@ -6,6 +6,7 @@ from __future__ import division
 from __future__ import print_function
 
 from caffe2.python import brew
+from caffe2.python.modeling.initializers import pFP16Initializer
 '''
 Utility for creating ResNets
 See "Deep Residual Learning for Image Recognition" by He, Zhang et. al. 2015
@@ -35,6 +36,8 @@ class ResNetBuilder():
             in_filters,
             out_filters,
             weight_init=("MSRAFill", {}),
+            WeightInitializer=pFP16Initializer,
+            BiasInitializer=pFP16Initializer,
             kernel=kernel,
             stride=stride,
             pad=pad,
@@ -119,6 +122,8 @@ class ResNetBuilder():
                 input_filters,
                 output_filters,
                 weight_init=("MSRAFill", {}),
+                WeightInitializer=pFP16Initializer,
+                BiasInitializer=pFP16Initializer,
                 kernel=1,
                 stride=(1 if down_sampling is False else 2),
                 no_bias=self.no_bias,
@@ -180,6 +185,8 @@ class ResNetBuilder():
                 input_filters,
                 num_filters,
                 weight_init=("MSRAFill", {}),
+                WeightInitializer=pFP16Initializer,
+                BiasInitializer=pFP16Initializer,
                 kernel=1,
                 stride=(1 if down_sampling is False else 2),
                 no_bias=self.no_bias,
@@ -228,6 +235,8 @@ def create_resnet50(
         num_input_channels,
         64,
         weight_init=("MSRAFill", {}),
+        WeightInitializer=pFP16Initializer,
+        BiasInitializer=pFP16Initializer,
         kernel=conv1_kernel,
         stride=conv1_stride,
         pad=3,
@@ -281,12 +290,17 @@ def create_resnet50(
 
     # Final dimension of the "image" is reduced to 7x7
     last_out = brew.fc(
-        model, final_avg, 'last_out_L{}'.format(num_labels), 2048, num_labels
+        model, final_avg, 'last_out_L{}'.format(num_labels), 2048, num_labels,
+        WeightInitializer=pFP16Initializer,
+        BiasInitializer=pFP16Initializer,
     )
 
     if no_loss:
         return last_out
 
+    # float16 --> float32
+    last_out = model.HalfToFloat(last_out, 'last_out_fp16')
+
     # If we create model for training, use softmax-with-loss
     if (label is not None):
         (softmax, loss) = model.SoftmaxWithLoss(
@@ -309,7 +323,9 @@ def create_resnet_32x32(
     '''
     # conv1 + maxpool
     brew.conv(
-        model, data, 'conv1', num_input_channels, 16, kernel=3, stride=1
+        model, data, 'conv1', num_input_channels, 16, kernel=3, stride=1,
+        WeightInitializer=pFP16Initializer,
+        BiasInitializer=pFP16Initializer,
     )
     brew.spatial_bn(
         model, 'conv1', 'conv1_spatbn', 16, epsilon=1e-3, is_test=is_test
diff --git a/train/resnet50/resnet50_trainer.py b/train/resnet50/resnet50_trainer.py
index 11f7439..2412305 100644
--- a/train/resnet50/resnet50_trainer.py
+++ b/train/resnet50/resnet50_trainer.py
@@ -59,10 +59,15 @@ def AddImageInput(model, reader, batch_size, img_size):
         std=128.,
         scale=256,
         crop=img_size,
-        mirror=1
+        mirror=1,
+        output_type='float16',
+        use_gpu_transform=True,
+        # output_type=core.DataType.FLOAT16,
     )
 
     data = model.StopGradient(data, data)
+    print(data.Net().__class__)
+    print(label.meta)
 
 
 def SaveModel(args, train_model, epoch):
@@ -270,15 +275,24 @@ def Train(args):
     def add_optimizer(model):
         stepsz = int(30 * args.epoch_size / total_batch_size / num_shards)
         optimizer.add_weight_decay(model, args.weight_decay)
-        optimizer.build_sgd(
+        opt = optimizer.build_multi_precision_sgd(
             model,
             args.base_learning_rate,
-            momentum=0.9,
+            momentump=0.9,
             nesterov=1,
             policy="step",
             stepsize=stepsz,
             gamma=0.1
         )
+        # optimizer.build_sgd(
+        #     model,
+        #     args.base_learning_rate,
+        #     momentump=0.9,
+        #     nesterov=1,
+        #     policy="step",
+        #     stepsize=stepsz,
+        #     gamma=0.1
+        # )
 
     # Input. Note that the reader must be shared with all GPUS.
     reader = train_model.CreateDB(
@@ -458,5 +472,5 @@ def main():
     Train(args)
 
 if __name__ == '__main__':
-    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
+    workspace.GlobalInit(['caffe2', '--caffe2_log_level=-10000'])
     main()
