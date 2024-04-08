# import libraries
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pylab as plt
import matplotlib.lines
import matplotlib.collections
import seaborn as sns
sns.set()
import datetime
import os
import arviz as az
az.style.use("arviz-whitegrid")
az.rcParams["stats.hdi_prob"] = 0.95
import pymc as pm
from fastprogress.fastprogress import force_console_behavior
master_bar, progress_bar = force_console_behavior()
print(f"Running on PyMC v{pm.__version__}")

df = pd.read_csv("/Users/Documents/TMDU/COVID19/BaseData/NumberOfDeathByPrefecture.csv", header = 0, index_col = 0)
df.mean().sort_values()
df.drop("ALL", axis = "columns", inplace = True)
df = df.iloc[0:685, :]

# to time-series data
df.index = pd.to_datetime(df.index)

# duration
start = pd.Period("2020-01-16", freq = "D")
end = pd.Period("2021-11-30", freq = "D")

# NPIs and ES
GoTo_start = datetime.date(2020, 7, 22)
GoToEat_start = datetime.date(2020, 10, 1)
SE1_start = datetime.date(2020,4,7)
SE1_end = datetime.date(2020,5,25)
SE2_start = datetime.date(2021,1,8)
SE2_end = datetime.date(2021,3,21)
SE3_start = datetime.date(2021,4,25)
SE3_end = datetime.date(2021,9,30)
school_closure_start = datetime.date(2020, 3, 2)
school_closure_end = datetime.date(2020, 4, 5)
GoTo_end = datetime.date(2020, 12, 28)
GoToEat_end = datetime.date(2020, 11, 24)
anl_start = datetime.date(2020, 1, 16)
anl_end = datetime.date(2021, 11, 30)

GoTo_start - anl_start# 188
GoToEat_start - anl_start# 259
GoTo_end - anl_start# 347
GoToEat_end - anl_start# 313

GoTo_start_day = 188
GoToEat_start_day = 259
GoTo_end_day = 347
GoToEat_end_day = 313

SE1_start-anl_start# 82
SE1_end-anl_start# 130
SE2_start-anl_start# 358
SE2_end-anl_start# 430
SE3_start-anl_start# 465
SE3_end-anl_start# 623

school_closure_start-anl_start# 46
school_closure_end-anl_start# 80

SE1_start_day = 82
SE1_end_day = 130
SE2_start_day = 358
SE2_end_day = 430
SE3_start_day = 465
SE3_end_day = 623

school_closure_start_day = 46
school_closure_end_day = 80

date = pd.date_range("2020-01-16", "2021-11-30")
N = len(date)

prefecture_idx, prefectures = pd.factorize(df.columns, sort = False)
coords = {"Prefecture":prefectures, "Date":date}
p = len(prefecture_idx)

# Prefecture adjustment
    # SE1
        # start
df_adjust_se1s = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
            # Tokyo, Kanagawa, Saitama, Chiba, Osaka, Hyogo, Fukuoka
df_adjust_se1s.loc[:, ["Tokyo", "Kanagawa", "Saitama", "Chiba", "Osaka", "Hyogo", "Fukuoka"]] = SE1_start_day
            # Other area
not_incolumns_se1s = df_adjust_se1s.columns[~df_adjust_se1s.columns.isin(["Tokyo", "Kanagawa", "Saitama", "Chiba", "Osaka", "Hyogo", "Fukuoka"])]
df_adjust_se1s.loc[:, not_incolumns_se1s] = 91# 91 = datetime.date(2020,4,16)-anl_start

        # end
df_adjust_se1e = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
            # Hokkaido, Tokyo, Kanagawa, Saitama, Chiba
df_adjust_se1e.loc[:, ["Hokkaido", "Tokyo", "Kanagawa", "Saitama", "Chiba"]] = SE1_end_day
            # Osaka, Hyogo
df_adjust_se1e.loc[:, ["Osaka", "Hyogo"]] = 126# 126 = datetime.date(2020,5,21)-anl_start
            # Other area
not_incolumns_se1e = df_adjust_se1e.columns[~df_adjust_se1e.columns.isin(["Hokkaido", "Tokyo", "Kanagawa", "Saitama", "Chiba", "Osaka", "Hyogo"])]
df_adjust_se1e.loc[:, not_incolumns_se1e] = 119# 119 = datetime.date(2020,5,14)-anl_start

    # SE2
        # start
df_adjust_se2s = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
            # Tokyo, Saitama, Chiba, Kanagawa
df_adjust_se2s.loc[:, ["Tokyo", "Saitama", "Chiba", "Kanagawa"]] = SE2_start_day
            # Aichi, Osaka, Hyogo, Fukuoka
df_adjust_se2s.loc[:, ["Aichi", "Osaka", "Hyogo", "Fukuoka"]] = 364# 364 = datetime.date(2021, 1,14)-anl_start
not_incolumns_se2s = df_adjust_se2s.columns[~df_adjust_se2s.columns.isin(["Tokyo", "Saitama", "Chiba", "Kanagawa", "Aichi", "Osaka", "Hyogo", "Fukuoka"])]
df_adjust_se2s.loc[:, not_incolumns_se2s] = 2*N
        # end
df_adjust_se2e = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
datetime.date(2021, 2,7)-anl_start
            # Tokyo, Kanagawa, Saitama, Chiba
df_adjust_se2e.loc[:, ["Tokyo", "Kanagawa", "Saitama", "Chiba"]] = SE2_end_day
            # Aichi, Osaka, Hyogo, Fukuoka
df_adjust_se2e.loc[:, ["Aichi", "Osaka", "Hyogo", "Fukuoka"]] = 409# 409 = datetime.date(2021,2,28)-anl_start
not_incolumns_se2e = df_adjust_se2e.columns[~df_adjust_se2e.columns.isin(["Tokyo", "Saitama", "Chiba", "Kanagawa", "Aichi", "Osaka", "Hyogo", "Fukuoka"])]
            # Other area
df_adjust_se2e.loc[:, not_incolumns_se2e] = 0

    # SE3 wave 1
        # start
df_adjust_se3s1 = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
            # Hokkaido
df_adjust_se3s1.loc[:, ["Hokkaido"]] = 486# 486: datetime.date(2021, 5, 16)-anl_start
datetime.date(2021, 5, 16)-anl_start
            # Tokyo, Osaka, Hyogo
df_adjust_se3s1.loc[:, ["Tokyo", "Osaka", "Hyogo"]] = 465# 465: datetime.date(2021, 4, 25)-anl_start
datetime.date(2021, 4, 25)-anl_start
            # Aichi, Fukuoka
df_adjust_se3s1.loc[:, ["Aichi", "Fukuoka"]] = 482# 482 = datetime.date(2021, 5, 12)-anl_start
datetime.date(2021, 5, 12)-anl_start
            # Okinawa
df_adjust_se3s1.loc[:, ["Okinawa"]] = 493# datetime.date(2021, 5, 23)-anl_start
datetime.date(2021, 5, 23)-anl_start
            # Other area
not_incolumns_se3s1 = df_adjust_se3s1.columns[~df_adjust_se3s1.columns.isin(["Hokkaido", "Tokyo", "Aichi", "Kyoto", "Osaka", "Hyogo", "Okayama", "Hiroshima", "Fukuoka", "Okinawa"])]
df_adjust_se3s1.loc[:, not_incolumns_se3s1] = 2*N
        # end
df_adjust_se3e1 = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
            # Hokkaido, Tokyo, Aichi, Osaka, Hyogo, Fukuoka
df_adjust_se3e1.loc[:, ["Hokkaido", "Tokyo", "Aichi", "Osaka", "Hyogo", "Fukuoka"]] = 521# 521: datetime.date(2021, 6, 20)-anl_start
datetime.date(2021, 6, 20)-anl_start
            # Okinawa
df_adjust_se3e1.loc[:, ["Okinawa"]] = SE3_end_day
            # Other area
not_incolumns_se3e1 = df_adjust_se3e1.columns[~df_adjust_se3e1.columns.isin(["Hokkaido", "Tokyo", "Aichi", "Osaka", "Hyogo", "Fukuoka", "Okinawa"])]
df_adjust_se3e1.loc[:, not_incolumns_se3e1] = 0

    # SE3 wave 2
        # start
df_adjust_se3s2 = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
            # Hokkaido, Aichi
df_adjust_se3s2.loc[:, ["Hokkaido", "Aichi"]] = 589# 598 = datetime.date(2021, 8, 27)-anl_start
datetime.date(2021, 8, 27)-anl_start
            # Tokyo
df_adjust_se3s2.loc[:, "Tokyo"] = 543# datetime.date(2021, 7, 12)-anl_start
datetime.date(2021, 7, 12)-anl_start
            # Saitama, Chiba, Kanagawa, Osaka
df_adjust_se3s2.loc[:, ["Chiba", "Saitama", "Kanagawa", "Osaka"]] = 564# datetime.date(2021, 8, 2)-anl_start
datetime.date(2021, 8, 2)-anl_start
            # Hyogo, Fukuoka
df_adjust_se3s2.loc[:, ["Hyogo", "Fukuoka"]] = 582# 582 = datetime.date(2021, 8, 20)-anl_start
datetime.date(2021, 8, 20)-anl_start
            # Other area
not_incolumns_se3s2 = df_adjust_se3s2.columns[~df_adjust_se3s2.columns.isin(["Hokkaido", "Tokyo", "Chiba", "Saitama", "Kanagawa", "Osaka", "Hyogo", "Aichi", "Fukuoka"])]
df_adjust_se3s2.loc[:, not_incolumns_se3s2] = 2*N
        # end
df_adjust_se3e2 = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
            # Hokkaido, Saitama, Chiba, Tokyo, Kanagawa, Aichi, Osaka, Hyogo, Fukuoka, Okinawa
df_adjust_se3e2.loc[:, ["Hokkaido", "Saitama", "Chiba", "Tokyo", "Kanagawa", "Aichi", "Osaka", "Hyogo", "Fukuoka", "Okinawa"]] = 623# 623: datetime.date(2021, 9, 30)-anl_start
datetime.date(2021, 9, 30)-anl_start
            # Other area
not_incolumns_se3e2 = df_adjust_se3e2.columns[~df_adjust_se3e2.columns.isin(["Hokkaido", "Saitama", "Chiba", "Tokyo", "Kanagawa", "Aichi", "Osaka", "Hyogo", "Fukuoka", "Okinawa"])]
df_adjust_se3e2.loc[:, not_incolumns_se3e2] = 0

    # School Closure
        # start
df_adjust_scs = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
df_adjust_scs.loc[:, :] = 46# 46 = datetime.date(2020, 3, 2)-anl_start
datetime.date(2020, 3, 2)-anl_start
        # end
df_adjust_sce = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
df_adjust_sce.loc[:, :] = 80# 80 = datetime.date(2020, 4, 5)-anl_start
datetime.date(2020, 4, 5)-anl_start

    # GoToTravel
        # start
df_adjust_travels = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
not_incolumns_travel= df_adjust_travels.columns[~df_adjust_travels.columns.isin(["Tokyo"])]
df_adjust_travels.loc[:, not_incolumns_travel] = GoTo_start_day
df_adjust_travels.loc[:, "Tokyo"] = GoToEat_start_day
        # end
df_adjust_travele = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
df_adjust_travele.loc[:, :] = GoTo_end_day
    # GoToEat
        # start
df_adjust_eats = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
df_adjust_eats.loc[:, :] = GoToEat_start_day
df_adjust_eate = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
df_adjust_eate.loc[:, :] = GoToEat_end_day

b = np.tile(np.arange(0, N), (p, 1)).T

se2_target_prefectures = ["Tokyo", "Kanagawa", "Saitama", "Chiba", "Aichi", "Osaka", "Hyogo", "Fukuoka"]
se2_is_target_prefecture = np.isin(prefectures, se2_target_prefectures)

se3_target_prefectures = ["Hokkaido", "Tokyo", "Aichi", "Kyoto", "Osaka", "Hyogo", "Okayama", "Hiroshima", "Fukuoka", "Okinawa"]
se3_is_target_prefecture = np.isin(prefectures, se3_target_prefectures)

# death model
with pm.Model(coords = coords) as hierarchical_death_model:
    # Prior Distributions
        # Prefecture-specifit R_0
    R_0 = df.apply(lambda x: [x.mean()] * len(x))
        # Mean NPI/ES effectiveness
    para_sigma = pm.HalfStudentT("para_sigma", nu = 3, sigma = 0.04)
    para_mean_es = pm.AsymmetricLaplace("para_mean_es", b = 10, kappa = 2.0, mu = 0)
    para_mean_npi = pm.AsymmetricLaplace("para_mean_npi", b = 10, kappa = 0.5, mu = 0)

    se2_is_target_prefecture_tensor = pm.math.constant(se2_is_target_prefecture)
    se3_is_target_prefecture_tensor = pm.math.constant(se3_is_target_prefecture)
    mu_delay = pm.Normal("mu_delay", mu = 21.82, sigma = 1.01)# mean: infection to death
    dispersion_delay = pm.Normal("dispersion_delay", mu = 14.26, sigma = 5.18)# dispersion: infection to death
    delay_goto = pm.NegativeBinomial("delay_goto", mu = mu_delay, alpha = dispersion_delay)
    delay_eat = pm.NegativeBinomial("delay_eat", mu = mu_delay, alpha = dispersion_delay)
    delay_se1 = pm.NegativeBinomial("delay_se1", mu = mu_delay, alpha = dispersion_delay)
    delay_se2 = pm.NegativeBinomial("delay_se2", mu = mu_delay, alpha = dispersion_delay)
    delay_se2_updated = pm.Deterministic("delay_se2_updated", pm.math.switch(se2_is_target_prefecture_tensor, delay_se2, 0))# detailed adjustment
    delay_se3 = pm.NegativeBinomial("delay_se3", mu = mu_delay, alpha = dispersion_delay)
    delay_se3_updated = pm.Deterministic("delay_se3_updated", pm.math.switch(se3_is_target_prefecture_tensor, delay_se3, 0))# detailed adjustment
    delay_sc = pm.NegativeBinomial("delay_sc", mu = mu_delay, alpha = dispersion_delay)
    # Observation Noise Dispersion Parameter
    psi = pm.HalfNormal("psi", sigma = 5)
    # ES and NPI parameters
    es1 = pm.Normal("es1", mu = para_mean_es, sigma = para_sigma, dims = "Prefecture")
    es2 = pm.Normal("es2", mu = para_mean_es, sigma = para_sigma, dims = "Prefecture")
    se1 = pm.Normal("se1", mu = para_mean_npi, sigma = para_sigma, dims = "Prefecture")
    se2 = pm.Normal("se2", mu = para_mean_npi, sigma = para_sigma, dims = "Prefecture")
    se2_updated = pm.Deterministic("se2_updated", pm.math.switch(se2_is_target_prefecture_tensor, se2, 0), dims = "Prefecture")# detailed adjustment
    se3_1 = pm.Normal("se3_1", mu = para_mean_npi, sigma = para_sigma, dims = "Prefecture")
    se3_1_updated = pm.Deterministic("se3_1_updated", pm.math.switch(se3_is_target_prefecture_tensor, se3_1, 0), dims = "Prefecture")# detailed adjustment
    se3_2 = pm.Normal("se3_2", mu = para_mean_npi, sigma = para_sigma, dims = "Prefecture")
    se3_2_updated = pm.Deterministic("se3_2_updated", pm.math.switch(se3_is_target_prefecture_tensor, se3_2, 0), dims = "Prefecture")# detailed adjustment
    sc = pm.Normal("sc", mu = para_mean_npi, sigma = para_sigma, dims = "Prefecture")
    # ES and NPIs
        # ES
            # GoToTravel
    scondition_goto_travel = (delay_goto < b-df_adjust_travels)
    econdition_goto_travel = (delay_goto >= b-df_adjust_travele)
    npi1 = pm.math.switch(scondition_goto_travel & econdition_goto_travel, es1, 0)
            # GoToEat
    scondition_goto_eat = (delay_eat < b-df_adjust_eats)
    econdition_goto_eat = (delay_eat >= b-df_adjust_eate)
    npi2 = pm.math.switch(scondition_goto_eat & econdition_goto_eat, es2, 0)
        # NPIs
            # SE1
    scondition_se1 = (delay_se1 < b-df_adjust_se1s)
    econdition_se1 = (delay_se1 >= b-df_adjust_se1e)
    npi3 = pm.math.switch(scondition_se1 & econdition_se1, se1, 0)
            # SE2
    scondition_se2 = (delay_se2_updated < b-df_adjust_se2s)
    econdition_se2 = (delay_se2_updated >= b-df_adjust_se2e)
    npi4 = pm.math.switch(scondition_se2 & econdition_se2, se2_updated, 0)
            # SE3 wave 1
    scondition_se31 = (delay_se3_updated < b-df_adjust_se3s1)
    econdition_se31 = (delay_se3_updated >= b-df_adjust_se3e1)
    npi5 = pm.math.switch(scondition_se31 & econdition_se31, se3_1_updated, 0)
    npi5_updated = pm.Deterministic("npi5_updated", pm.math.switch(is_target_prefecture_tensor, npi5, 0))
            # SE3 wave 2
    scondition_se32 = (delay_se3_updated < b-df_adjust_se3s2)
    econdition_se32 = (delay_se3_updated >= b-df_adjust_se3e2)
    npi6 = pm.math.switch(scondition_se32 & econdition_se32, se3_2_updated, 0)
    npi6_updated = pm.Deterministic("npi6_updated", pm.math.switch(is_target_prefecture_tensor, npi6, 0))
            # SC
    scondition_sc = (delay_sc < b-df_adjust_scs)
    econdition_sc = (delay_sc >= b-df_adjust_sce)
    npi7 = pm.math.switch(scondition_sc & econdition_sc, sc, 0)
        # Combine all npis and es
    npi = pm.Deterministic("npi", npi1+npi2+npi3+npi4+npi5_updated+npi6_updated+npi7)
    cnpi = pm.Deterministic("cnpi", npi3+npi4+npi5_updated+npi6_updated+npi7)
    # Infection Model
    R_e = pm.Deterministic("R_e", pm.math.exp(R_0)*pm.math.exp(-npi))
    R_c = pm.Deterministic("R_c", pm.math.exp(R_0)*pm.math.exp(-cnpi))
    # prefectures
    death = pm.NegativeBinomial("death", mu = R_e, alpha = psi, observed = df.iloc[:, prefecture_idx])

    with hierarchical_death_model:
    idata = pm.sample(draws = 5000,
                    tune = 2500,
                    chains = 4,
                    cores = 12,
                    init = "advi",
                    n_init = 2500,
                    target_accept = 0.99,
                    return_inferencedata = True
                    )

# Summary statistics
    # delay
az.summary(idata, var_names = ["delay_goto"], round_to = 2)
az.summary(idata, var_names = ["delay_eat"], round_to = 2)
az.summary(idata, var_names = ["delay_se1"], round_to = 2)
az.summary(idata, var_names = ["delay_se2"], round_to = 2)
az.summary(idata, var_names = ["delay_se3"], round_to = 2)
az.summary(idata, var_names = ["delay_sc"], round_to = 2)
    # ES & NPIs
az.summary(idata, var_names = ["es1"], round_to = 2)
az.summary(idata, var_names = ["es2"], round_to = 2)
az.summary(idata, var_names = ["se1"], round_to = 2)
az.summary(idata, var_names = ["se2_updated"], round_to = 2)
az.summary(idata, var_names = ["se3_1_updated"], round_to = 2)
az.summary(idata, var_names = ["se3_2_updated"], round_to = 2)
az.summary(idata, var_names = ["sc"], round_to = 2)
    # mean of all prefectures
az.summary(idata, var_names = ["es1"])["mean"].mean(), az.summary(idata, var_names = ["es1"])["hdi_2.5%"].mean(),az.summary(idata, var_names = ["es1"])["hdi_97.5%"].mean()
az.summary(idata, var_names = ["es2"])["mean"].mean(), az.summary(idata, var_names = ["es2"])["hdi_2.5%"].mean(),az.summary(idata, var_names = ["es2"])["hdi_97.5%"].mean()
az.summary(idata, var_names = ["se1"])["mean"].mean(), az.summary(idata, var_names = ["se1"])["hdi_2.5%"].mean(),az.summary(idata, var_names = ["se1"])["hdi_97.5%"].mean()
az.summary(idata, var_names = ["se2_updated"])["mean"].mean(), az.summary(idata, var_names = ["se2_updated"])["hdi_2.5%"].mean(),az.summary(idata, var_names = ["se2_updated"])["hdi_97.5%"].mean()
az.summary(idata, var_names = ["se3_1_updated"])["sd"].mean(), az.summary(idata, var_names = ["se3_1_updated"])["hdi_2.5%"].mean(),az.summary(idata, var_names = ["se3_1_updated"])["hdi_97.5%"].mean()
az.summary(idata, var_names = ["se3_2_updated"])["mean"].mean(), az.summary(idata, var_names = ["se3_2_updated"])["hdi_2.5%"].mean(),az.summary(idata, var_names = ["se3_2_updated"])["hdi_97.5%"].mean()
az.summary(idata, var_names = ["sc"])["mean"].mean(), az.summary(idata, var_names = ["sc"])["hdi_2.5%"].mean(),az.summary(idata, var_names = ["sc"])["hdi_97.5%"].mean()

# plot
    # ES1
az.plot_forest(idata, var_names=["es1"],
               combined=False, colors=["C1"], figsize=(11.5, 10),
               hdi_prob=0.95, r_hat=False)
plt.axvline(x=0, color="red", linestyle="dashed")
plt.title("Go To Travel \n"r"$\leftarrow$""increased COVID-19 death \n did not increase COVID-19 death"r"$\rightarrow$",
          loc="center", fontsize=24)
plt.show()
    # ES2
az.plot_forest(idata, var_names=["es2"],
               combined=False, colors=["C2"], figsize=(11.5, 10),
               hdi_prob=0.95, r_hat=False)
plt.axvline(x=0, color="red", linestyle="dashed")
plt.title("Go To Eat \n"r"$\leftarrow$""increased COVID-19 death \n did not increase COVID-19 death"r"$\rightarrow$",
          loc="center", fontsize=24)
plt.show()
    # SE1
az.plot_forest(idata, var_names=["se1"],
               combined=False, colors=["C3"], figsize=(11.5, 10),
               hdi_prob=0.95, r_hat=False)
plt.axvline(x=0, color="red", linestyle="dashed")
plt.title("SE 1 \n"r"$\leftarrow$""failed to reduce COVID-19 death \n reduced COVID-19 death"r"$\rightarrow$",
          loc="center", fontsize=24)
plt.show()
    # SE2
az.plot_forest(idata, var_names=["se2_updated"],
               combined=False, colors=["C4"], figsize=(11.5, 10),
               hdi_prob=0.95, r_hat=False)
plt.axvline(x=0, color="red", linestyle="dashed")
plt.title("SE 2 \n"r"$\leftarrow$""failed to reduce COVID-19 death \n reduced COVID-19 death"r"$\rightarrow$",
          loc="center", fontsize=24)
plt.show()
    # SE3 wave 1
az.plot_forest(idata, var_names=["se3_1_updated"],
               combined=False, colors=["C5"], figsize=(11.5, 10),
               hdi_prob=0.95, r_hat=False)
plt.axvline(x=0, color="red", linestyle="dashed")
plt.title("SE 3 (wave 1) \n"r"$\leftarrow$""failed to reduce COVID-19 death \n reduced COVID-19 death"r"$\rightarrow$",
          loc="center", fontsize=24)
plt.show()
    # SE3 wave 2
az.plot_forest(idata, var_names=["se3_2_updated"],
               combined=False, colors=["C6"], figsize=(11.5, 10),
               hdi_prob=0.95, r_hat=False)
plt.axvline(x=0, color="red", linestyle="dashed")
plt.title("SE 3 (wave 2) \n"r"$\leftarrow$""failed to reduce COVID-19 death \n reduced COVID-19 death"r"$\rightarrow$",
          loc="center", fontsize=24)
plt.show()
# SC
az.plot_forest(idata, var_names=["sc"],
               combined=False, colors=["C7"], figsize=(11.5, 10),
               hdi_prob=0.95, r_hat=False)
plt.axvline(x=0, color="red", linestyle="dashed")
plt.title("SC \n"r"$\leftarrow$""failed to reduce COVID-19 death \n reduced COVID-19 death"r"$\rightarrow$",
          loc="center", fontsize=24)
plt.show()

# for sensitivity analysis
# death model
with pm.Model(coords = coords) as sa_hierarchical_death_model:
    # Prior Distributions
        # Prefecture-specifit R_0
    R_0 = df.apply(lambda x: [x.mean()] * len(x))
        # Mean NPI/ES effectiveness
    para_sigma = pm.HalfStudentT("para_sigma", nu = 3, sigma = 0.04)
    para_mean_es = pm.AsymmetricLaplace("para_mean_es", b = 10, kappa = 1.0, mu = 0)
    para_mean_npi = pm.AsymmetricLaplace("para_mean_npi", b = 10, kappa = 1.0, mu = 0)

    se2_is_target_prefecture_tensor = pm.math.constant(se2_is_target_prefecture)
    se3_is_target_prefecture_tensor = pm.math.constant(se3_is_target_prefecture)
    mu_delay = pm.Normal("mu_delay", mu = 21.82, sigma = 1.01)# mean: infection to death
    dispersion_delay = pm.Normal("dispersion_delay", mu = 14.26, sigma = 5.18)# dispersion: infection to death
    delay_goto = pm.NegativeBinomial("delay_goto", mu = mu_delay, alpha = dispersion_delay)
    delay_eat = pm.NegativeBinomial("delay_eat", mu = mu_delay, alpha = dispersion_delay)
    delay_se1 = pm.NegativeBinomial("delay_se1", mu = mu_delay, alpha = dispersion_delay)
    delay_se2 = pm.NegativeBinomial("delay_se2", mu = mu_delay, alpha = dispersion_delay)
    delay_se2_updated = pm.Deterministic("delay_se2_updated", pm.math.switch(se2_is_target_prefecture_tensor, delay_se2, 0))# detailed adjustment
    delay_se3 = pm.NegativeBinomial("delay_se3", mu = mu_delay, alpha = dispersion_delay)
    delay_se3_updated = pm.Deterministic("delay_se3_updated", pm.math.switch(se3_is_target_prefecture_tensor, delay_se3, 0))# detailed adjustment
    delay_sc = pm.NegativeBinomial("delay_sc", mu = mu_delay, alpha = dispersion_delay)
    # Observation Noise Dispersion Parameter
    psi = pm.HalfNormal("psi", sigma = 5)
    # ES and NPI parameters
    es1 = pm.Normal("es1", mu = para_mean_es, sigma = para_sigma, dims = "Prefecture")
    es2 = pm.Normal("es2", mu = para_mean_es, sigma = para_sigma, dims = "Prefecture")
    se1 = pm.Normal("se1", mu = para_mean_npi, sigma = para_sigma, dims = "Prefecture")
    se2 = pm.Normal("se2", mu = para_mean_npi, sigma = para_sigma, dims = "Prefecture")
    se2_updated = pm.Deterministic("se2_updated", pm.math.switch(se2_is_target_prefecture_tensor, se2, 0), dims = "Prefecture")# detailed adjustment
    se3_1 = pm.Normal("se3_1", mu = para_mean_npi, sigma = para_sigma, dims = "Prefecture")
    se3_1_updated = pm.Deterministic("se3_1_updated", pm.math.switch(se3_is_target_prefecture_tensor, se3_1, 0), dims = "Prefecture")# detailed adjustment
    se3_2 = pm.Normal("se3_2", mu = para_mean_npi, sigma = para_sigma, dims = "Prefecture")
    se3_2_updated = pm.Deterministic("se3_2_updated", pm.math.switch(se3_is_target_prefecture_tensor, se3_2, 0), dims = "Prefecture")# detailed adjustment
    sc = pm.Normal("sc", mu = para_mean_npi, sigma = para_sigma, dims = "Prefecture")
    # ES and NPIs
        # ES
            # GoToTravel
    scondition_goto_travel = (delay_goto < b-df_adjust_travels)
    econdition_goto_travel = (delay_goto >= b-df_adjust_travele)
    npi1 = pm.math.switch(scondition_goto_travel & econdition_goto_travel, es1, 0)
            # GoToEat
    scondition_goto_eat = (delay_eat < b-df_adjust_eats)
    econdition_goto_eat = (delay_eat >= b-df_adjust_eate)
    npi2 = pm.math.switch(scondition_goto_eat & econdition_goto_eat, es2, 0)
        # NPIs
            # SE1
    scondition_se1 = (delay_se1 < b-df_adjust_se1s)
    econdition_se1 = (delay_se1 >= b-df_adjust_se1e)
    npi3 = pm.math.switch(scondition_se1 & econdition_se1, se1, 0)
            # SE2
    scondition_se2 = (delay_se2_updated < b-df_adjust_se2s)
    econdition_se2 = (delay_se2_updated >= b-df_adjust_se2e)
    npi4 = pm.math.switch(scondition_se2 & econdition_se2, se2_updated, 0)
            # SE3 wave 1
    scondition_se31 = (delay_se3_updated < b-df_adjust_se3s1)
    econdition_se31 = (delay_se3_updated >= b-df_adjust_se3e1)
    npi5 = pm.math.switch(scondition_se31 & econdition_se31, se3_1_updated, 0)
    npi5_updated = pm.Deterministic("npi5_updated", pm.math.switch(is_target_prefecture_tensor, npi5, 0))
            # SE3 wave 2
    scondition_se32 = (delay_se3_updated < b-df_adjust_se3s2)
    econdition_se32 = (delay_se3_updated >= b-df_adjust_se3e2)
    npi6 = pm.math.switch(scondition_se32 & econdition_se32, se3_2_updated, 0)
    npi6_updated = pm.Deterministic("npi6_updated", pm.math.switch(is_target_prefecture_tensor, npi6, 0))
            # SC
    scondition_sc = (delay_sc < b-df_adjust_scs)
    econdition_sc = (delay_sc >= b-df_adjust_sce)
    npi7 = pm.math.switch(scondition_sc & econdition_sc, sc, 0)
        # Combine all npis and es
    npi = pm.Deterministic("npi", npi1+npi2+npi3+npi4+npi5_updated+npi6_updated+npi7)
    cnpi = pm.Deterministic("cnpi", npi3+npi4+npi5_updated+npi6_updated+npi7)
    # Infection Model
    R_e = pm.Deterministic("R_e", pm.math.exp(R_0)*pm.math.exp(-npi))
    R_c = pm.Deterministic("R_c", pm.math.exp(R_0)*pm.math.exp(-cnpi))
    # prefectures
    death = pm.NegativeBinomial("death", mu = R_e, alpha = psi, observed = df.iloc[:, prefecture_idx])

with sa_hierarchical_death_model:
    saidata = pm.sample(draws = 5000,
                    tune = 2500,
                    chains = 4,
                    cores = 12,
                    init = "advi",
                    n_init = 2500,
                    target_accept = 0.99,
                    return_inferencedata = True
                    )
# Summary statistics
    # delay
az.summary(saidata, var_names = ["delay_goto"], round_to = 2)
az.summary(saidata, var_names = ["delay_eat"], round_to = 2)
az.summary(saidata, var_names = ["delay_se1"], round_to = 2)
az.summary(saidata, var_names = ["delay_se2"], round_to = 2)
az.summary(saidata, var_names = ["delay_se3"], round_to = 2)
az.summary(saidata, var_names = ["delay_sc"], round_to = 2)
    # ES & NPIs
az.summary(saidata, var_names = ["es1"], round_to = 2)
az.summary(saidata, var_names = ["es2"], round_to = 2)
az.summary(saidata, var_names = ["se1"], round_to = 2)
az.summary(saidata, var_names = ["se2_updated"], round_to = 2)
az.summary(saidata, var_names = ["se3_1_updated"], round_to = 2)
az.summary(saidata, var_names = ["se3_2_updated"], round_to = 2)
az.summary(saidata, var_names = ["sc"], round_to = 2)
    # mean of all prefectures
az.summary(saidata, var_names = ["es1"])["mean"].mean(), az.summary(saidata, var_names = ["es1"])["hdi_2.5%"].mean(),az.summary(saidata, var_names = ["es1"])["hdi_97.5%"].mean()
az.summary(saidata, var_names = ["es2"])["mean"].mean(), az.summary(saidata, var_names = ["es2"])["hdi_2.5%"].mean(),az.summary(saidata, var_names = ["es2"])["hdi_97.5%"].mean()
