# Train data correlogram

train_csv: train_data/train_sp500_us10y.csv
rows: 14734
features: sp500, DGS10
max_lag: 252

Selected ACF values:
- sp500 return: lag1=-0.014755, lag5=-0.009608, lag10=0.011753, lag20=0.003635, lag40=-0.006275, lag252=-0.008198
- sp500 absolute_return: lag1=0.262842, lag5=0.290453, lag10=0.244330, lag20=0.180678, lag40=0.144160, lag252=0.037751
- sp500 squared_return: lag1=0.229941, lag5=0.228031, lag10=0.135831, lag20=0.066932, lag40=0.057831, lag252=0.003069
- DGS10 return: lag1=0.060637, lag5=0.032911, lag10=0.004483, lag20=0.019709, lag40=-0.003035, lag252=-0.008673
- DGS10 absolute_return: lag1=0.253672, lag5=0.260699, lag10=0.226683, lag20=0.227039, lag40=0.205717, lag252=0.147422
- DGS10 squared_return: lag1=0.190865, lag5=0.196960, lag10=0.139394, lag20=0.136659, lag40=0.185939, lag252=0.092300
