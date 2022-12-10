#BATCHES=(8 16 32)
#EPOCHS=(25 50 100)

BATCHES=(32)
OPTIMIZERS = ('optimizer') # 'adam' 'sgd' 'adadelta' 'adagrad')

for bs in "${BATCHES[@]}"; do 
    for o in "${OPTIMIZERS[@]}"; do 
        echo $e $bs
        python run.py --model paper --data ./processed_images --batch_size $bs -- optimizer $o
    done 
done 

