# weights/

Trained weights of the five cross-validation fold models (ConvNeXt-Tiny, four heads; ~111 MB each) exceed the GitHub file-size limit and are distributed as assets of GitHub release v1.0.0 (https://github.com/HazarGozgoz/ife-deep-learning/releases/tag/v1.0.0):

- `fold1_best_ife_kesin_384px_pw10.pth` … `fold5_best_ife_kesin_384px_pw10.pth`

Load with `egitim_ve_degerlendirme.py --model <path>` (see the script's `--help`). Each file is a PyTorch state_dict for the model defined in `src/egitim_ve_degerlendirme.py`.
