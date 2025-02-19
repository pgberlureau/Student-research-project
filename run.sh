python dataset.py --size 10000 --width 32 --height 32 --dir dataset

python train.py --model_path models/model.pt --dataset_path dataset --curves_path curves/curves.png
python train.py --model_path models/variant.pt --variant --dataset_path dataset --curves_path curves/curves_variant.png

python sampling.py --width 32 --height 32 --model_path models/model.pt --num_samples 10 --samples_path samples/samples32 --stats_path stats/stats32.txt
python sampling.py --width 48 --height 48 --model_path models/model.pt --num_samples 10 --samples_path samples/samples48 --stats_path stats/stats48.txt
python sampling.py --width 64 --height 64 --model_path models/model.pt --num_samples 10 --samples_path samples/samples64 --stats_path stats/stats64.txt

python sampling.py --width 32 --height 32 --model_path models/variant.pt --variant --num_samples 10 --samples_path samples/samples_variant32 --stats_path stats/stats_variant32.txt
python sampling.py --width 48 --height 48 --model_path models/variant.pt --variant --num_samples 10 --samples_path samples/samples_variant48 --stats_path stats/stats_variant48.txt
python sampling.py --width 64 --height 64 --model_path models/variant.pt --variant --num_samples 10 --samples_path samples/samples_variant64 --stats_path stats/stats_variant64.txt