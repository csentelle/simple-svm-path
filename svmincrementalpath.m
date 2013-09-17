% Copyright (c) 2013, Christopher Sentelle
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions 
% are met:
% 
% Redistributions of source code must retain the above copyright notice, 
% this list of conditions and the following disclaimer.
% Redistributions in binary form must reproduce the above copyright notice, 
% this list of conditions and the following disclaimer in the documentation 
% and/or other materials provided with the distribution.
% Neither the name of the organization nor the names of its contributors 
% may be used to endorse or promote products derived from this software 
% without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
% "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, 
% BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
% FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
% COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
% INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
% HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
% STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING 
% IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
% POSSIBILITY OF SUCH DAMAGE.

function [lambdavec] = svmincrementalpath(P, T, ktype, params, rho, outfile) %#codegen
% [lambdavec] = svmincrementalpath(P, T, ktype, params, rho, outfile) 
% Computes the entire regularization path based upon:
%
% Sentelle, Christopher, Anagnostopoulos, Georgios C., Georgiopoulos,
% Michael, "A Simple Method for Solving the SVM Regularization Path for 
% Semi-definite Kernels," IEEE Transactions on Neural Networks and 
% Learning Systems, Submitted.
%
% This has been implemented for comparative study of regularization path
% methods only. A list of breakpoints is returned by the method and 
% additional stats, logging is stored in the outfile. If there is
% sufficient demand, a future version will include kernel caching for
% larger datasets as well as a cross validation routine for evaluating
% performance at a subset of breakpoints and returning trained SVM for
% optimal $\lambda$ value. There are also plans to fully support MATLAB
% coder for generating a compiled MEX library as well as standalone API 
% in the future. 
%
% Input arguments
% P - N x M matrix of training examples with M features
% T - N x 1 vector of labels {-1, 1}
% ktype - 'rbf' or 'linear' (for now, will add more in future)
% params - empty ([]) for linear kernel, [lambda] for RBF kernel
% rho - Relative distance to artificial data points for unequal class size
%       initialization. Recommend setting of 0.1. See journal publication.
% outfile - output file for debug trace.

coder.extrinsic('num2str', 'disp');

if (nargin == 6 && ~isempty(outfile)),
    fd = fopen(outfile,'w');
else
    fd = -1;
end
if (nargin <= 4)
    rho = .1;
end

tkernel = tic;
tcpu = tic;

K = zeros(length(T),length(T));

switch (ktype)
    case 'rbf'    
        lambda = params(1);
        for i = 1:length(T),
             K(i,:) = exp(-lambda * sum((P(i,:)'*ones(1,size(P',2)) - P').^2,1));
        end
    case 'linear'
        K = (P*P');
    otherwise
        error('Unrecognized kernel type');
end

tkernel = toc(tkernel);

pex = sum(T==1);
mex = sum(T==-1);
auxmode = false;
idxaux = [];

% Note that we are storing the entire kernel matrix. This should not be
% done for very large datasets, future version will include kernel caching.
% Also, we are not taking advantage of the fact that the auxiliary
% variables are repeats and the kernel matrix really only needs to grow by
% 1 row/column, if at all if we decide to track the data point implicitly.
% See IEEE publication (if/when published...).

% Auxiliary variable generation for N_+ != N_- or unequal class sizes
taux = tic;
if (pex ~= mex),
    auxmode = true;
    idxaux = length(T)+1:length(T) + abs(pex - mex);
    
    g = K*T;
    
    if (pex > mex),
        K = [K, g * ones(1,pex-mex) * -rho; 
             ones(pex-mex,1) * g' * -rho, rho^2 * sum(g.*T) * ones(pex-mex,pex-mex)];
        T = [T; -ones(pex-mex,1)];
    elseif (mex > pex)
        
        K = [K, g * ones(1,mex-pex) * rho; 
             ones(mex-pex,1) * g' * rho, rho^2 * sum(g.*T) * ones(mex-pex,mex-pex)];
        T = [T; ones(mex-pex,1)];
    end
end
taux = toc(taux);

tk2 = tic;
Q = K.*(T*T');
tk2 = toc(tk2);
tkernel = tkernel + tk2;

g = K*T;

[bp, ip] = max(g(T==1));
[bn, in] = min(g(T==-1));

ipa = find(T==1);
ina = find(T==-1);

lambda = (bp - bn)/2;
b0 = -(bp + bn)/(bp - bn);

a0 = b0 * lambda;
ip = ipa(ip);
in = ina(in);

idx_nb = zeros(length(T),1);

idxl = ones(length(T),1);
idxr = zeros(length(T),1);

idx_nb(1) = ip;
idx_nb(2) = in;
num_nb = 2;

idxl(ip) = 0;
idxl(in) = 0;

alpha = ones(length(T),1);

f = 1/lambda * (K*(alpha.*T) + a0);

if (fd > 0),
    fprintf(fd, 'Initial\r\n');
    fprintf(fd, 'lambda = %f\r\n', lambda);
    fprintf(fd, 'a0 = %f\r\n', a0);
    fprintf(fd, 'E = %d\r\n', [ip, in]);
end

% % disp(['lambda = ',num2str(lambda)]);
% % disp(['a0 = ', num2str(a0)]);
% % disp(['E = ', num2str(find(idx_nb==1)')]);

m_R = 0;

% Start the m_R matrix
m_R = UpdateCholesky(m_R, Q(ip,ip), T(ip));
m_R = UpdateCholesky(m_R, Q([ip in],[ip in]), T([ip in])');

iter = 1;
nrepeatiter = 0;
nauxiter = 0;
tauxiter = 0;
nrepeat = 0;

tinnercpu = tic;
tdisp = 0;
max_nb = 0;
ave_nb = 0;

lambdavec = [];

while (lambda > 1e-3 && ~isempty(idxl)),

%     if (~checkKKT(alpha, T, f, idxl, idxr)),
%         disp('algorithm failure');
% %         Find the offending point and move to idx_nb, etc.
%     end
    
    
    if (~auxmode),
        max_nb = max(max_nb, num_nb);
        ave_nb = ave_nb + num_nb;
        lambdavec = [lambdavec; lambda];
    end
    
    
    ba = solveSub(m_R, Q(idx_nb(1:num_nb),idx_nb(1:num_nb)), T(idx_nb(1:num_nb))');
    b = ba(1:end-1);
    b0 = ba(end);

    
    lambdat = zeros(num_nb,1);
    idxnbt = idx_nb(1:num_nb);
    e = 0;
    lambdat(:) = 0;
    lambdat(b < -e) = (1 + lambda * b(b < -e) - alpha(idxnbt(b < -e))) ./(b(b < -e));
    lambdat(b > e) = (lambda * b(b > e) - alpha(idxnbt(b > e)))./(b(b > e));

    h = K(:,idx_nb(1:num_nb))*(b.*T(idx_nb(1:num_nb))) + b0;
    
    lambdaf = zeros(length(idxl),1);
    for i = 1:length(f),
        %
        % Handle the potential 0/0 issue for points on the margin
        % but in L or R with no significant direction change noted
        %
        if (idxl(i) || idxr(i)),

            if (abs(T(i) - h(i) ) < 1e-6)
                lambdaf(i) = 0;               
            elseif (T(i)*(f(i) - h(i)) < -1e-6 && idxl(i)),
                lambdaf(i) = 0;
            elseif (T(i)*(f(i) - h(i)) > 1e-6 && idxr(i)),
                lambdaf(i) = 0;
            else
                lambdaf(i) = lambda * (f(i) - h(i))/(T(i) - h(i));
            end
        end
    end;


    % Guard against numerical errors
    lambdaf((lambdaf - lambda)/lambda > 1e-6) = 0;

    % Note that the max function returns the min index in case of ties,
    % which supports the Bland pivot rule.
    [lambdatmax, idxa] = max(lambdat);    
    [lambdafmax, idxf] = max(lambdaf);
    
    lambdap = max(lambdatmax, lambdafmax);
    

    
    if (lambdap > 1e-3), 
                     
        
        % Perform incremental updates
        alpha(idx_nb(1:num_nb)) = alpha(idx_nb(1:num_nb)) - (lambda - lambdap) * b;
        a0 = a0 - (lambda - lambdap) * b0;
        f = lambda/lambdap * (f - h) + h;            
        
        %
        % Determine which event occurred, in case of tie, ensure minimum
        % index is chosen per minimum pivot rule.
        %
        if (lambdatmax > lambdafmax || ...
            (abs(lambdatmax - lambdafmax)/lambdafmax < 1e-9 && idx_nb(idxa) < idxf)),
            
            if( alpha(idx_nb(idxa)) < 1e-6),
                idxr(idx_nb(idxa)) = 1;
            else
                idxl(idx_nb(idxa)) = 1;
            end

            m_R = DownDateCholesky(m_R, T(idx_nb(1:num_nb))', idxa);
            
            idxtmp = idx_nb(idxa);
            
            for i = idxa:num_nb-1,
                idx_nb(i) = idx_nb(i+1);
            end
            num_nb = num_nb - 1;
            
            
        else
            
           

            num_nb = num_nb + 1;
            idx_nb(num_nb) = idxf;
            
            if (idxl(idxf) == 1), 
                idxl(idxf) = 0;
            else
                idxr(idxf) = 0;
            end
            
            [m_R] = UpdateCholesky(m_R, Q(idx_nb(1:num_nb), idx_nb(1:num_nb)), T(idx_nb(1:num_nb))');
            
            % This can occur if numerical issues begin to arise, especially
            % when $\lambda_j - \lambda_{j+1} < tolerance$. As reported,
            % this was not an issue as long as the minimum value of
            % $\lambda$ was greather than 10^{-4}. 
            if (min(abs(diag(m_R))) < 1e-5)
                disp('Stopping for Singular Matrix');
                lambdap = 1e-6;
            end
        end

        dtime = tic;
        
        if (fd > 0),
            
            fprintf(fd,'====================================================\r\n');
            fprintf(fd,'lambda = : %f\r\n', lambda);
            fprintf(fd,'a0 =     : %f\r\n', a0);
            fprintf(fd,'iter =   : %d\r\n', iter);

            if (lambdatmax > lambdafmax)
                fprintf(fd,'Removed %d\r\n', idxtmp);
            else
                fprintf(fd,'Added %d\r\n', idxf);
            end
            fprintf(fd, 'E Size = : %d\r\n', num_nb);
            
        end
 
        
%         disp('===================================================');
%         disp(['lambda =      :  ', num2str(lambda)]);
%         disp(['num nb = :', num2str(num_nb)]);
%         disp(['E =           :  ', num2str(idx_nb(1:num_nb)')]);
%         disp(['L =           :  ', num2str(find(idxl==1)')]);
%         disp(['R =           :  ', num2str(find(idxr==1)')]);
%         disp(['a0 =          :  ', num2str(a0)]);
%         disp(['iter =        :  ', num2str(iter)]);


        tdisp = tdisp + toc(dtime);
        
        iter = iter + 1;
        if (~auxmode),
            if (abs(lambda - lambdap)/lambdap < 1e-9), 
                nrepeatiter = nrepeatiter + 1;
                nrepeat = nrepeat + 1;
                if (nrepeat > 1000),
                    %
                    % Hopefully this never occurs based upon theoretical
                    % results! If there are 1000 duplicates of a single
                    % data point, this event could be hit anyway, in which
                    % case the threshold should be adjusted.
                    %
                    if (fd > 0)
                        fprintf(fd, '=========================\r\n');
                        fprintf(fd, 'Cycling stop\r\n');
                        fprintf(fd, '=========================\r\n');
                    end
                    error('Stopped for cycling');
                end
            else
                nrepeat = 0;
            end
        end
        
        if (auxmode),
            if (max(alpha(idxaux)) > 1e-6),
                nauxiter = nauxiter + 1;
                tauxiter = toc(tinnercpu) - tdisp;
            else
                auxmode = 0;
                if (fd > 0), fprintf(fd, 'End Auxiliary Mode =====================\r\n'); end
            end
        end
    end

    lambda = lambdap;


end 


tcpu = toc(tcpu);
tinnercpu = toc(tinnercpu);

fprintf('==============================================================\n');
fprintf('Total algo cpu time =                        :  %f\n', tcpu-tdisp);
fprintf('Inner algo cpu time =                        :  %f\n', tinnercpu - tdisp);
fprintf('Kernel computation cpu time =                :  %f\n', tkernel);
fprintf('Auxiliary variable generation cpu time =     :  %f\n', taux);
fprintf('Auxiliary initialization cpu time =          :  %f\n', tauxiter);
fprintf('File writing cpu time =                      :  %f\n', tdisp);
fprintf('Num main events (excluding auxiliary) =      :  %d\n', iter - nauxiter);
fprintf('Num repeat lambda events (excluding aux)  =  :  %d\n', nrepeatiter);
fprintf('Num auxiliary variable events =              :  %d\n', nauxiter);
fprintf('Max E size =                                 :  %d\n', max_nb);
fprintf('Avg E size =                                 :  %f\n', ave_nb / (iter - nauxiter));
fprintf('==============================================================\n');

if (fd > 0),
    fprintf(fd, '==============================================================\r\n');
    fprintf(fd, 'Total algo cpu time =                        :  %f\r\n', tcpu-tdisp);
    fprintf(fd, 'Inner algo cpu time =                        :  %f\r\n', tinnercpu - tdisp);
    fprintf(fd, 'Kernel computation cpu time =                :  %f\r\n', tkernel);
    fprintf(fd, 'Auxiliary variable generation cpu time =     :  %f\r\n', taux);
    fprintf(fd, 'Auxiliary initialization cpu time =          :  %f\r\n', tauxiter);
    fprintf(fd, 'File writing cpu time =                      :  %f\r\n', tdisp);
    fprintf(fd, 'Num main events (excluding auxiliary) =      :  %d\r\n', iter - nauxiter);
    fprintf(fd, 'Num repeat lambda events (excluding aux)  =  :  %d\r\n', nrepeatiter);
    fprintf(fd, 'Num auxiliary variable events =              :  %d\r\n', nauxiter);
    fprintf(fd, 'Max E size =                                 :  %d\r\n', max_nb);
    fprintf(fd, 'Avg E size =                                 :  %f\r\n', ave_nb / (iter - nauxiter));
    fprintf(fd, '==============================================================\r\n');
    fclose(fd);
end

function valid = checkKKT(alpha, T, f, idxl, idxr)

ekkt = 1e-6;
idx = find((idxr &   f.*T < 1-ekkt) | ...
           (idxl & f.*T > 1 + ekkt) | ...
           (alpha > ekkt & alpha < 1-ekkt & abs(f.*T-1) > ekkt) | ...
            (idxl & alpha < 1-ekkt) | ...
            (idxr & alpha > ekkt),1);

if (~isempty(idx) || abs(sum(T.*alpha)) > ekkt),
    disp(num2str(idx));
    [T(idx).*f(idx) alpha(idx)]
    valid = false;
else
    valid = true;
end

function [m_R, valid] = UpdateCholesky(m_R, Q, T)
%  
%   Q, here, is the portion of the larger Q for the current non-bound support
%   vectors. The last row/column represents the row/column to be added. T
%   is the set of labels for the non-bound support vectors and the last
%   entry represents the entry to be added. 
% 
%  Update the Cholesky factorization by solving the following
%
%  R^T*r = -y_1 * y_n * Z^T * Q * e_1 + Z^T * q
%  r^T*r + rho^2 = e_1^T * Q * e_1 - 2 * y_1 * y_n * e_1^T * q + sigma
%

   valid = true;    
   Z = [-T(1) * T(2:end-1); eye(length(T) - 2) ] ;
   if (length(T) == 1), 
       m_R = sqrt(Q);
   elseif (length(T) == 2),
       m_R = sqrt([-T(1)*T(2) 1] * Q * [-T(1)*T(2); 1]);
   else
           
       q = Q(1:end-1,end);
       sigma = Q(end,end);

       r = m_R' \(-T(1)*T(end) * Z' * Q(1:end-1,1) + Z' * q) ;
       rho = sqrt(Q(1,1) - 2 * T(1) * T(end) * q(1) + sigma - r'*r);

           m_R = [m_R, r; 
                   zeros(1,size(m_R,1)), rho];
   end

   
function m_R = DownDateCholesky(m_R, T, idx)

%
%   Here, we downdate the Cholesky factorization by removing the 
%   row/column, indexed by idx. There are two cases to consider. 
%   (1) if 2 <= idx <= end, remove the row, perform Givens rotations, and 
%   convert back to upper triangular and return the reduced R
%   (2) if idx = 1, apply the transformation A to R, then convert to upper
%   triangular and reduced the R to the new size. In this case, the
%   transform A is defined as 
%       [-y2 * [y3, ..., yn] 1; 
%               I            0];
%   

    if (idx > 1)

        m_R(:,idx-1) = [];

        for i = idx - 1: size(m_R, 1) - 1,
            m_R(i:i+1,:) = givens(m_R(i, i), m_R(i+1, i)) * m_R(i:i+1,:);
        end

        m_R = m_R(1:end-1,:);

    else

        A = [-T(2) * T(3:end), 1; 
             eye(length(T(3:end))), zeros(length(T(3:end)),1)];

        m_R = m_R * A;

        for i = 1:size(m_R,1) - 1,
            m_R(i:i+1,:) = givens(m_R(i, i), m_R(i+1, i)) * m_R(i:i+1,:);
        end

        m_R = m_R(1:end-1,1:end-1);        

    end
    

function out = solveSub(m_R, Q, y)
 
            
    % Solve using the NULL space method.
    % Solve the following in sequence:
    %  1. y'Yhy = r for hy
    %      hy = y(1)r
    %      Y = [1; 0; 0; ...]
    %  2. Z'QZhz = Z'(q-QYhy) for hz
    %      R'R = -Z'QZ
    %      -R'Rhz = Z'(q-QYhy)
    %      rhs = Z'(q-QYhy)
    %      -R'Rhz = rhs
    %      -R'x = rhs
    %      Rhz = x
    %  3. h = Zhz + Yhy
    %  4. Y'Qh + Y'yg = Y'q for g
    %     Q(1,:)h + y(1)g = q(1)     
    %     
    if (length(y) > 1)
        
        Z = [-y(1)*y(2:end); eye(length(y)-1)];
        
        rhs = Z'*ones(length(y),1);
        
        hz = m_R'\rhs;
        hz = m_R\hz;
                
        h = Z*hz;                        
        g = y(1)*(1 - Q(1,:)*h);
        
        out = [h; g];
    else
        % Just solve the following
        % 1. h = y(1)*r
        % 2. g = y(1)*(q - Q*h)
        %         

        out = [0; y(1)];
    end
