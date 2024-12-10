for x in {0..13}; do
    wd=$(echo "$x * 0.00001 + 0.00001" | bc -l)
    python run/wd.py adamw $wd
done
