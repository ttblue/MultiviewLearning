function mm_handle = mm_rbf_fourierfeatures(m, d)

  W = normrnd(0,1,m,d);
  h = rand(1,d)*2*pi;

  mm_handle = @mm_rbf;

  function xhat = mm_rbf(x, a)
    xhat = mean(cos(x*(W/a)+repmat(h,size(x,1),1)))/sqrt(d)*sqrt(2);
  end

end
