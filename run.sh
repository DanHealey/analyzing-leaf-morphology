#BATCHES=(8 16 32)
#EPOCHS=(25 50 100)

BATCHES=(32)
EPOCHS=(1)

for bs in "${BATCHES[@]}"; do 
    for e in "${EPOCHS[@]}"; do 
        echo $e $bs
        python run.py --model paper --data ./processed_images --epochs $e --batch_size $bs
    done 
done 

