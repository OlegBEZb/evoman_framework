for (( ENEMY=1; ENEMY<=8; ENEMY+=1 )); do
    for (( LAUNCH=1; LAUNCH<=10; LAUNCH+=1 )); do
        echo "**************************************"
        echo "********** ENEMY $ENEMY LAUNCH $LAUNCH **********"
        python experiment.py $LAUNCH $ENEMY
    done
done