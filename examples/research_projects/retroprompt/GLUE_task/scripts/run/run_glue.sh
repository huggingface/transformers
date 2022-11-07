# running arguments
TYPE="prompt"
MODEL="roberta-large"
MODEL_NAME="roberta_large"
TASK=SST-2  # task name (SST-2 mr cr MNLI QNLI QQP ...)

case $TASK in
    SST-2)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{'0':'terrible','1':'great'}"
        MAX_LENGTH=128
        ;;
    mr)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        MAX_LENGTH=128
        ;;
    QQP)
        TEMPLATE=*cls**sent_0**mask*,*sent_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        MAX_LENGTH=128
        ;;
    MNLI)
        TEMPLATE=*cls**sent_0*?*mask*,*sent_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        MAX_LENGTH=220
        ;;
    QNLI)
        TEMPLATE=*cls**sent_0*?*mask*,*sent_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        MAX_LENGTH=256
        ;;
    cr)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        MAX_LENGTH=128
        ;;
    few_nerd)
        TEMPLATE=*cls**sent_0*sent_1*_is*mask*.*sep+*
        MAPPING="{0:['person','artist'],1:['person','actor'],2:['art','writtenart'],3:['person','director'],4:['person','other'],5:['organization','other'],6:['organization','company'],7:['organization','sportsteam'],8:['organization','sportsleague'],9:['product','car'],10:['event','protest'],11:['organization','government'],12:['other','biologything'],13:['location','GPE'],14:['location','other'],15:['person','athlete'],16:['art','broadcastprogram'],17:['product','other'],18:['building','other'],19:['product','weapon'],20:['building','airport'],21:['building','sportsfacility'],22:['person','scholar'],23:['art','music'],24:['event','other'],25:['other','language'],26:['other','chemicalthing'],27:['art','film'],28:['building','hospital'],29:['other','law'],30:['product','airplane'],31:['location','railway'],32:['person','soldier'],33:['location','mountain'],34:['organization','education'],35:['organization','media'],36:['product','software'],37:['location','island'],38:['location','bodiesofwater'],39:['building','library'],40:['other','astronomything'],41:['person','politician'],42:['building','hotel'],43:['product','game'],44:['other','award'],45:['event','sportsevent'],46:['organization','showorganization'],47:['other','educationaldegree'],48:['building','theater'],49:['other','disease'],50:['event','election'],51:['organization','politicalparty'],52:['other','currency'],53:['event','attack'],54:['product','ship'],55:['building','restaurant'],56:['other','livingthing'],57:['art','other'],58:['event','disaster'],59:['organization','religion'],60:['other','medical'],61:['location','park'],62:['other','god'],63:['product','food'],64:['product','train'],65:['art','painting']}"
        MAX_LENGTH=256
        ;;
esac

# hyper parameters
BS=8
LR=1e-5         # learning rate (1e-5, 3e-5, 5e-5)

K=16            # kshot (16 4)
MAX_STEP=800    # total training steps
EVAL_STEP=80

REAL_BS=8       # batch size   (8, 4, 2)
GS=$(expr $BS / $REAL_BS)

SEED=13         # seed (13 21 42 87 100)
beta=0.0001     # beta in knn-train (0.0001-10)

DATA_DIR=data/training_data/k_shot/$TASK/$K-$SEED
OUTPUT_DIR=outputs/$TASK-$K-$SEED-$REAL_BS-$LR

CUDA_VISIBLE_DEVICES=1 python run.py \
  --task_name $TASK \
  --data_dir $DATA_DIR \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --do_predict \
  --prompt true \
  --template $TEMPLATE \
  --mapping $MAPPING \
  --model_name_or_path $MODEL \
  --few_shot_type $TYPE \
  --num_k $K \
  --max_seq_length $MAX_LENGTH \
  --per_device_train_batch_size $REAL_BS \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps $GS \
  --learning_rate $LR \
  --max_steps $MAX_STEP \
  --logging_steps $EVAL_STEP \
  --eval_steps $EVAL_STEP \
  --num_train_epochs 0 \
  --output_dir $OUTPUT_DIR \
  --seed $SEED \
  --model_type "roberta" \
  --use_demo \
  --demo_num 1 \
  --demo_topk 8 \
  --train_with_knn \
  --only_train_knn \
  --beta $beta \
  --knn_topk 16 \
  --knn_mode \
  --knn_lambda 0.2


# rm $OUTPUT_DIR/pytorch_model.bin $OUTPUT_DIR/vocab.json $OUTPUT_DIR/config.json $OUTPUT_DIR/merges.txt $OUTPUT_DIR/special_tokens_map.json
