for i in `seq 0 99`
do
  echo "i = $i";
  file_name=$(printf logs/%03d.log $i);\
  python squeezenet/sample.py > $file_name
done
