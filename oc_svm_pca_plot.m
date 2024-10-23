figure

plot(oc_dimension_400, svm_train_time, 'ob-','DisplayName','OC-SVM Train Time');
% plot(oc_dimension_1200, svm_train_time, 'ob-','DisplayName','OC-SVM Train Time');
hold on;

plot(oc_dimension_400, pca_train_time, 'sr-','DisplayName','PCA Train Time');
% plot(oc_dimension_1200, pca_train_time, 'sr-','DisplayName','PCA Train Time');
hold on;

grid on;

legend('show',Location='northwest');

xlabel('Dimension');
ylabel('Train Time');
