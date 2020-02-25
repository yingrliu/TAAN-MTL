clear all
Lambda = 0.0:0.1:0.4;
NoVMTL = [83.68 85.60 84.17 83.58 83.53];
Euclidean = [83.68 85.60 84.17 83.58 83.53;
             85.54 85.82 86.16 86.69 85.48;
             85.91 85.63 85.04 84.79 86.26;
             87.19 86.82 86.91 87.03 85.67;
             86.63 86.97 85.35 87.03 85.85];
TraceNorm = [83.68 85.60 84.17 83.58 83.53;
             79.51 80.78 82.40 80.72 75.93;
             75.87 76.37 75.65 75.16 77.02;
             74.47 76.77 75.53 77.99 76.18;
             75.96 76.43 75.75 75.22 75.93];
Cosine = [83.68 85.60 84.17 83.58 83.53;
          83.74 83.15 79.82 84.17 83.33;
          80.66 77.24 81.00 82.46 82.59;
          76.21 82.25 83.40 80.69 80.72;
          87.25 78.45 83.27 80.53 85.60];
      
%%
NoVMTL_M = mean(NoVMTL, 2);
Euclidean_M = mean(Euclidean, 2);
TraceNorm_M = mean(TraceNorm, 2);
Cosine_M = mean(Cosine, 2);
NoVMTL_std = std(NoVMTL, 0, 2);
Euclidean_std = std(Euclidean, 0, 2);
TraceNorm_std = std(TraceNorm, 0, 2);
Cosine_std = std(Cosine, 0, 2);
hold on;
box on;
errorbar(Lambda, Cosine_M, Cosine_std, '-*', 'linewidth', 2.5, 'CapSize',15)
errorbar(Lambda, Euclidean_M, Euclidean_std, '-*', 'linewidth', 2.5, 'CapSize',15)
errorbar(Lambda, TraceNorm_M, TraceNorm_std, '-*', 'linewidth', 2.5, 'CapSize',15)
errorbar(Lambda(1), NoVMTL_M, NoVMTL_std, '-*', 'linewidth', 2.5, 'CapSize',15)
legend('$\mathcal{L}_{Cos}$', '$\mathcal{L}_{2N}$', '$\mathcal{L}_{TN}$', 'c=0$', 'Interpreter','latex')
xlabel('$\lambda$','Interpreter','latex')
ylabel('Test Accuracy')
set(gca,'XTick',0.0:0.1:0.4)

