rm -rf squeezenet/logs
mkdir squeezenet/logs

for i in `seq 0 99`
do
  echo "i = $i";
  file_name=$(printf squeezenet/logs/%03d.log $i);\
  python squeezenet/sample.py > $file_name
done


rm -rf bvlc_googlenet/logs
mkdir bvlc_googlenet/logs

for i in `seq 0 99`
do
  echo "i = $i";
  file_name=$(printf bvlc_googlenet/logs/%03d.log $i);\
  python bvlc_googlenet/sample.py > $file_name
done
