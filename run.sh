#BATCHES=(8 16 32)
#EPOCHS=(25 50 100)

BATCHES=(8 16 32)
declare -a OPTIMIZERS=("adam" "sgd" "adadelta" "adagrad")
declare -a LOSSES=("binary_crossentropy" "categorical_crossentropy" "sparse_categorical_crossentropy")

for l in "${LOSSES[@]}"; do 
    for bs in "${BATCHES[@]}"; do 
        for o in "${OPTIMIZERS[@]}"; do 
            echo $bs $o $l
            python run.py --model paper --data ./processed_images --batch_size $bs --optimizer $o --loss $l
        done 
    done 
done

