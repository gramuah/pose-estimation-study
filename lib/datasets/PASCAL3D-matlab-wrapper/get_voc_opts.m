function classes_list = get_voc_opts(path)
tmp_path = pwd;

voc_code_path = 'VDPM';
pascal3d_devkit = [path, filesep , voc_code_path];
cd(pascal3d_devkit);
try
  addpath(pascal3d_devkit);
  pascal_init;
catch
  rmpath(voc_code_path);
  cd(tmp_path);
  error(sprintf('VOCcode directory not found under %s', path));
end
cd(tmp_path);

classes_list={'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor'};
