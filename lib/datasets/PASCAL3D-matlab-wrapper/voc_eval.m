function res = voc_eval(path, comp_id, test_set, data_path, vnum_train, output_dir, rm_res)

classes_list = get_voc_opts(path);

for i = 1:length(classes_list)
  cls = classes_list{i};
  res(i) = voc_eval_cls(data_path, test_set, cls, vnum_train, comp_id, output_dir, rm_res);
end

fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Results:\n');
fprintf('\nDetection\n');
aps = [res(:).ap]';
fprintf('%.1f\n', aps * 100);
fprintf('%.1f\n', mean(aps) * 100);
fprintf('\nAverage Viewpoint Precision\n');
avp = [res.ap_auc]';
fprintf('%.1f\n', avp * 100);
fprintf('%.1f\n', mean(avp) * 100);
fprintf('~~~~~~~~~~~~~~~~~~~~\n');

function res = voc_eval_cls(data_path, data_set, cls, vnum_train, comp_id, output_dir, rm_res)

[recall, prec, acc, ap, ap_auc] = compute_pose_recall_precision_accuracy(data_path, comp_id, data_set, cls, vnum_train);
% ap_auc = xVOCap(recall, prec);

% force plot limits
% ylim([0 1]);
% xlim([0 1]);

print(gcf, '-djpeg', '-r0', ...
    [output_dir '/' cls '_pr.jpg']);

fprintf('!!! %s : %.4f %.4f\n', cls, ap, ap_auc);

res.recall = recall;
res.prec = prec;
res.ap = ap;
res.accuracy = acc;
res.ap_auc = ap_auc;

save([output_dir '/' cls '_pr.mat'], 'res');

if rm_res
  delete(res_fn);
end
