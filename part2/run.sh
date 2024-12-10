#python run/wd.py adam 0.00004
#python run/wd.py adam 0.00003
#python run/wd.py adam 0.00002
#python run/wd.py adam 0.00001
#python run/wd.py adam 0.000009
#python run/wd.py adam 0.000008
#python run/wd.py adam 0.000007
#python run/wd.py adam 0.000006
#python run/wd.py adam 0.000005
#python run/wd.py adam 0.000004

for x in {0..3}; do
    wd=$(echo "$x * 0.00001 + 0.00001" | bc -l)
    python run/wd.py adam $wd
done

for x in {0..5}; do
    wd=$(echo "$x * 0.000001 + 0.000004" | bc -l)
    python run/wd.py adam $wd
done