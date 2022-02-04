set -e
GIT_LFS_SKIP_SMUDGE=1
mkdir /root/.huggingface
echo -n ${5} > /root/.huggingface/token
git config --global credential.helper store
echo "\n\n\n\nauthenticated at HF with token"
n="${6:-5}"
echo "\n\n\n\nn set to ${n}"
echo "\n\n\n\ndownloading text data from dataset ${1}/${2} and concatenating"
python3 make_dataset.py $1 $2
echo "\n\n\n\ncalculating ngram model for n=${n}"
kenlm/build/bin/lmplz -o ${n} <"text.txt" > "${n}gram.arpa"
echo "\n\n\n\nadding EOS token to ngram model"
python3 fix_ngram_lm.py ${n}
echo "\n\n\n\nadding ngram model to hugging face model ${3}/${4} and pushing"
python3 add_ngram_and_push.py $3 $4 ${n}