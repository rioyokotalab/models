for i in `seq 0 99`
do
  echo "i = $i";
  file_name=$(printf logs/%03d.log $i);\
  python bvlc_googlenet.py > $name
done
