function VOCopts = get_voc_opts(path)

voc_code_path = 'PASCAL/VOCdevkit/VOCcode';
tmp = pwd;
cd(path);
try
  addpath(voc_code_path);
  VOCinit;
catch
  rmpath(voc_code_path);
  cd(tmp);
  error(sprintf('VOCcode directory not found under %s', path));
end
rmpath(voc_code_path);
cd(tmp);
