import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore",category=Warning)
pd.set_option("display.width",100)
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
import matplotlib
matplotlib.use("Qt5Agg")

pokemon = pd.read_csv("free_work/datasets/pokemon/pokemon.csv")
combat = pd.read_csv("free_work/datasets/pokemon/combats.csv")
tests = pd.read_csv("free_work/datasets/pokemon/tests.csv")


pokemon.columns = [col.replace(" ", "_").upper() for col in pokemon.columns]
pokemon = pokemon.rename(index=str, columns={"#": "ID"})
pokemon.loc[pokemon['NAME'].isnull(),"NAME"] = "Primeape"
pokemon[pokemon['NAME'].isnull()]

pokemon.head()
combat.head()
pokemon.isnull().sum()
pokemon.describe().T


#--------------------------------- Feature Engineering ----------------------------------------------

# Feature averages for each pokemon
pokemon["FEATURES_MEAN"] = round(pokemon.loc[:,["HP","ATTACK","DEFENSE","SP._ATK","SP._DEF","SPEED"]].mean(axis=1),2)
now = combat.groupby('Winner').count()

# Total matches for each pokemon
now["TOTAL_MATCHES"] = combat.groupby("First_pokemon")["Winner"].count() + combat.groupby("Second_pokemon")["Winner"].count()

# Total winnings for each pokemon
combat["Winner"].value_counts().sort_index()

# Win rates for each pokemon
now["WIN_RATE"] = round(combat["Winner"].value_counts().sort_index()/(now["TOTAL_MATCHES"])*100,2)

now.head()

pokemon = pd.merge(pokemon, now, left_on='ID', right_index = True, how='left')
pokemon.sort_values(by="WIN_RATE",ascending=False).head(10)

del pokemon["First_pokemon"]
del pokemon["Second_pokemon"]

# There are pokemons that don't have a win rate, I won't delete them because I'll deal with them while preparing the data
pokemon.isnull().sum()
pokemon.head()
pokemon.shape

# ---------------------------------- Data Analysis & Visualization -----------------------------------
# ANALYSIS 1

# TYPE_1 PERCENT IN TERMS OF WINNING RATE
fig, axs = plt.subplots(figsize=(14, 8))
axs = sns.barplot(x="TYPE_1",y="WIN_RATE",data=pokemon,errwidth=0,palette="magma").set_title('PERCENTAGE OF TYPE_1 BY WINNING RATE')
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
plt.axhline(np.mean(pokemon["WIN_RATE"]), color='white', linestyle='--',  linewidth=2, label='Average')
plt.xticks(rotation=60)
plt.legend()
plt.tight_layout()

# Orange Color
#sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'orange'})


# NUMBER OF TYPE_1
fig, axs = plt.subplots(figsize=(14, 8))
axs = sns.countplot(x="TYPE_1",data=pokemon,palette="magma").set_title('NUMBER OF TYPE_1')
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
plt.xticks(rotation=60)
plt.legend()
plt.tight_layout()


# TYPE_2 PERCENT IN TERMS OF WINNING RATE
fig, axs = plt.subplots(figsize=(14, 8))
axs = sns.barplot(x="TYPE_2",y="WIN_RATE",data=pokemon,errwidth=0,palette="magma").set_title('PERCENTAGE OF TYPE_2 BY WINNING RATE')
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
plt.axhline(np.mean(pokemon["WIN_RATE"]), color='white', linestyle='--',  linewidth=2, label='Average')
plt.xticks(rotation=60)
plt.legend()
plt.tight_layout()


# NUMBER OF TYPE_2
fig, axs = plt.subplots(figsize=(14, 8))
axs = sns.countplot(x="TYPE_2",data=pokemon,palette="magma").set_title('NUMBER OF TYPE_2')
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
plt.xticks(rotation=60)
plt.legend()
plt.tight_layout()


# When we look at the graph of type1 and type2, we can say that they are more dominant in terms of winning percentage (dragon, dark and flying).
pokemon.groupby("TYPE_1")["WIN_RATE"].mean().sort_values(ascending=False)
pokemon.groupby("TYPE_2")["WIN_RATE"].mean().sort_values(ascending=False)

# Other color options
"""
#palette="rocket"

fig, axs = plt.subplots(figsize=(14, 8))
axs = sns.barplot(x="TYPE_1",y="WIN_RATE",data=pokemon,errwidth=0,palette="rocket").set_title('TYPE_1')
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
plt.axhline(np.mean(pokemon["WIN_RATE"]), color='white', linestyle='--',  linewidth=2, label='Average')
plt.xticks(rotation=60)
plt.legend()
plt.tight_layout()

#palette="mako"

fig, axs = plt.subplots(figsize=(14, 8))
axs = sns.barplot(x="TYPE_1",y="WIN_RATE",data=pokemon,errwidth=0,palette="mako").set_title('TYPE_1')
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
plt.axhline(np.mean(pokemon["WIN_RATE"]), color='white', linestyle='--',  linewidth=2, label='Average')
plt.xticks(rotation=60)
plt.legend()
plt.tight_layout()

"""

"""
# PIE CHART
labels = pokemon['TYPE_1'].unique()
fig2, axs = plt.subplots(figsize=(14, 8))
plt.pie(pokemon["TYPE_1"].value_counts(),labels=labels, autopct='%.0f%%',shadow=True,explode=[0.08, 0.08, 0.08, 0.08, 0.08, 0.08,0.08, 0.08, 0.08, 0.08, 0.08, 0.08,0.08, 0.08, 0.08, 0.08, 0.08, 0.08],textprops=dict(color="w"), startangle = 90,
        wedgeprops= {"edgecolor":"black",
                     'linewidth': 3,
                     'antialiased': True})
sns.set_theme(font="arial",palette="mako")
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
plt.title('Percentage of TYPE_1')
# fig.update_traces(textposition='outside')
# fig.update_layout(uniformtext_minsize=12, uniformtext_mode='show')
plt.show()

pokemon["TYPE_1"].value_counts()

"""
# --------------------------------------------------------------------------------------------------
# ANALYSIS 2

# When we look at the top 10 Pokemon in terms of win rate, 50% of them are Mega Pokemon.
pokemon.sort_values(by="WIN_RATE",ascending=False).head(10)

# When we look at the top 10 Pokemon in terms of features_mean, 60% of them are Mega Pokemon.
pokemon.sort_values(by="FEATURES_MEAN",ascending=False).head(10)


cls_wr = pokemon[pokemon["NAME"].str.contains("Mega")].sort_values(by="WIN_RATE",ascending=False)
cls_wr["WR_BINS"] = pd.cut(x=cls_wr["WIN_RATE"],bins=[0, 50, 70, 80, 100],labels=['<50',"50-70","70-80","80>"])

# PIE CHART
labels = cls_wr["WR_BINS"].unique()
fig2, axs = plt.subplots(figsize=(14, 8))
plt.pie(cls_wr["WR_BINS"].value_counts(),labels=labels, autopct='%.0f%%',shadow=True,explode=[0.08, 0.08, 0.08,0.08],textprops=dict(color="black"), startangle = 90,
        wedgeprops= {"edgecolor":"black",
                     'linewidth': 3,
                     'antialiased': True})
sns.set_theme(font="arial",palette="magma")
sns.set(style="ticks", context="talk")
# plt.style.use("dark_background")
plt.title('PERCENTAGE OF MEGA POKEMONS BY WINNING RATE')
# fig.update_traces(textposition='outside')
# fig.update_layout(uniformtext_minsize=12, uniformtext_mode='show')
plt.show()
# Data shows us, the winning percentage of Pokemon with mega in its name is quite high.

# ----------------------------- Correlation Analysis & Pairplot & Kdeplot ------------------------------

pokemon.iloc[:,1:].corr(numeric_only=True)

plt.figure(figsize= (12,8))
list2_ = [0.00,.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.8,.85,.90,.95,1]
labels=['HP','ATCK', 'DEF', 'SP.ATCK', 'SP.DEF', 'SPEED', 'GEN','LEG', 'FEA_MN','TT_MATC','WIN_R']
cbar_kws = {"shrink":.8,'extend':'both',"ticks":list2_}
sns.heatmap(pokemon.iloc[:,1:].corr(numeric_only=True) ,linewidth=2.5,linecolor="k",cbar_kws=cbar_kws,cmap="Rocket",xticklabels=labels,yticklabels=labels)

#  Looking at the features with the highest correlation to winning which are speed,attack,sp_atck and features_mean.

# pairplot
list_ = ["WIN_RATE","FEATURES_MEAN","SPEED","SP._ATK","ATTACK"]
sns.pairplot(pokemon.loc[:,list_],height=2.5)

# kdeplot
g = sns.pairplot(pokemon.loc[:,list_], diag_kind="kde",height=2.5)
g.map_lower(sns.kdeplot, levels=4, color=".2")



# ----------------------------------- Preparation of Data --------------------------------------------
# I want to arrange the data before modeling and optimization so I will use the given data to predict match results

df = pokemon.copy()
df.head()
combat.head()
combat.shape
# attack sp.atck speed fe_mea win_ra

attack_fp = []
sp_atck_fp = []
speed_fp = []
fea_mea_fp = []
win_ra_fp = []
attack_sp = []
sp_atck_sp = []
speed_sp = []
fea_mea_sp = []
win_ra_sp = []

list_name = [attack_fp,sp_atck_fp,speed_fp,fea_mea_fp,win_ra_fp,attack_sp,sp_atck_sp,speed_sp,fea_mea_sp,win_ra_sp]


def preparer(col,col_name):

    for val in combat[col]:
        for id in df["ID"]:
            if id == val:
                list_name[i-1].append(df.loc[df["ID"] == id, col_name].values[0])

list_col = ["ATTACK","SP._ATK","SPEED","FEATURES_MEAN","WIN_RATE"]
for col in combat.iloc[:,:2]:
    for col_name in list_col:
        i +=1
        preparer(col,col_name)


pok_pok = pd.DataFrame(data={"First_pokemon":combat["First_pokemon"],
                             "fp_attack":attack_fp,
                             "fp_sp_attack":sp_atck_fp,
                             "fp_speed":speed_fp,
                             "fp_feature_mean":fea_mea_fp,
                             "fp_win_rate":win_ra_fp,
                             "Second_pokemon":combat["Second_pokemon"],
                             "sp_attack": attack_sp,
                             "sp_sp_attack": sp_atck_sp,
                             "sp_speed": speed_sp,
                             "sp_feature_mean": fea_mea_sp,
                             "sp_win_rate": win_ra_sp,
                             "Winner":combat["Winner"]})

pok_pok.head()
pok_pok.isnull().sum()
pok_pok.fillna(0,inplace=True)
# pok_pok2 = pok_pok.copy()
# If first pokemon win --> 0, If second pokemon win --> 1
pok_pok['Winner']=np.where(pok_pok['First_pokemon']==pok_pok['Winner'],0,1)
pok_pok.head()

# ----------------------------------------------- MODEL -------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, precision_score, recall_score, f1_score,accuracy_score,roc_auc_score


x = pok_pok.drop(["Winner"],axis=1)
y = pok_pok[["Winner"]]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

###########################
# Random Forest
###########################
rf_model = RandomForestClassifier()
rf_model.fit(x_train,y_train)

y_pred_rf = rf_model.predict(x_test)
acc_rf = accuracy_score(y_pred_rf,y_test) # 0.95

cm = confusion_matrix(y_pred_rf,y_test)
print(classification_report(y_pred_rf,y_test))
print(f"ROC AUC: {roc_auc_score(y_pred_rf, y_test)}, ACCURACY: {acc_rf}")

###########################
# XGBoost
###########################
xgboost_model = XGBClassifier()
xgboost_model.fit(x_train,y_train)

y_pred_xgb = xgboost_model.predict(x_test)
acc_xgb = round(accuracy_score(y_pred_xgb,y_test),2)


cm = confusion_matrix(y_pred_xgb,y_test)
print(classification_report(y_pred_xgb,y_test))
roc_auc_score(y_pred_xgb, y_test)

###########################
# LightGBM
###########################
lgb_model = lgb.LGBMClassifier(verbose=-1)
lgb_model.fit(x_train,y_train)

y_pred_lgb = lgb_model.predict(x_test)
acc_lgb = round(accuracy_score(y_pred_lgb,y_test),2)


cm = confusion_matrix(y_pred_lgb,y_test)
print(classification_report(y_pred_lgb,y_test))
roc_auc_score(y_pred_lgb, y_test)

# ------------------------------------------ H-P-Optimizasyonu ----------------------------------
from sklearn.model_selection import GridSearchCV

# RandomForestClassifier
rf_params = {"max_depth":[8,10],
             "max_features":[4,8,"auto"],
             "min_samples_split":[5,8,15],
             "n_estimators":[100,200,500]}

rf_best_grid = GridSearchCV(rf_model,rf_params,cv=5,n_jobs=-1,verbose=0).fit(x_train,y_train)
rf_best_grid.best_params_
"""
{'max_depth': 8,
 'max_features': 8,
 'min_samples_split': 5,
 'n_estimators': 100}
"""

rf_final = RandomForestClassifier().set_params(**rf_best_grid.best_params_).fit(x_train,y_train)
#rf_final = RandomForestClassifier(max_depth= 8,max_features=8,min_samples_split= 5,n_estimators=100).fit(x_train,y_train)
y_pred_rf_final = rf_final.predict(x_test)

print(f"Accuracy:  {(accuracy_score(y_pred_rf_final, y_test)):.2f}\nPrecision: {(precision_score(y_pred_rf_final, y_test)):.2f}\nRecall:    {(recall_score(y_pred_rf_final, y_test)):.2f}\nF1 score:  {(f1_score(y_pred_rf_final, y_test)):.2f}\nROC:       {(roc_auc_score(y_pred_rf_final, y_test)):.2f}")
acc_rf_final = accuracy_score(y_pred_rf_final, y_test) # 0.9411333333333334

cm=classification_report(y_pred_rf_final,y_test)

# -------------------------------------------------------------------------------------------------------
# XGBoost

xgboost_params = {"max_depth":[8,10],
             "learning_rate":[0.1,0.01],
             "colsample_bytree":[0.7,1],
             "n_estimators":[100,200,300]}

xgboost_best_grid = GridSearchCV(xgboost_model,xgboost_params,cv=5,n_jobs=-1,verbose=0).fit(x_train,y_train)
xgboost_best_grid.best_params_
"""
{'colsample_bytree': 0.7,
 'learning_rate': 0.1,
 'max_depth': 10,
 'n_estimators': 200}
"""

xgboost_final = XGBClassifier().set_params(**xgboost_best_grid.best_params_).fit(x_train,y_train)
# xgboost_final = XGBClassifier(colsample_bytree= 0.7,learning_rate=0.1,max_depth= 10,n_estimators=200).fit(x_train,y_train)
y_pred_xgb_final = xgboost_final.predict(x_test)

print(f"Accuracy:  {(accuracy_score(y_pred_xgb_final, y_test)):.2f}\nPrecision: {(precision_score(y_pred_xgb_final, y_test)):.2f}\nRecall:    {(recall_score(y_pred_xgb_final, y_test)):.2f}\nF1 score:  {(f1_score(y_pred_xgb_final, y_test)):.2f}\nROC:       {(roc_auc_score(y_pred_xgb_final, y_test)):.2f}")
acc_xgb_final = accuracy_score(y_pred_xgb_final, y_test) # 0.9510666666666666

cm=classification_report(y_pred_xgb_final,y_test)

# -------------------------------------------------------------------------------------------------------
# LightGBM

lgb_model = lgb.LGBMClassifier(verbose=-1)
lgbm_params = {"learning_rate":[0.1,0.01],
             "colsample_bytree":[0.7,1],
             "n_estimators":[100,200,500]
               }

lgbm_best_grid = GridSearchCV(lgb_model,lgbm_params,cv=5,n_jobs=-1,verbose=0).fit(x_train,y_train)
lgbm_best_grid.best_params_
"""
{'colsample_bytree': 0.7,
 'learning_rate': 0.1,
 'n_estimators': 500}
"""

lgbm_final = lgb.LGBMClassifier(verbose=-1).set_params(**lgbm_best_grid.best_params_).fit(x_train,y_train)
# lgbm_final = lgb.LGBMClassifier(colsample_bytree= 0.7,learning_rate=0.1,n_estimators=500,verbose=-1).fit(x_train,y_train)
y_pred_lgb_final = lgbm_final.predict(x_test)

print(f"Accuracy:  {(accuracy_score(y_pred_lgb_final, y_test)):.2f}\nPrecision: {(precision_score(y_pred_lgb_final, y_test)):.2f}\nRecall:    {(recall_score(y_pred_lgb_final, y_test)):.2f}\nF1 score:  {(f1_score(y_pred_lgb_final, y_test)):.2f}\nROC:       {(roc_auc_score(y_pred_lgb_final, y_test)):.2f}")
acc_lgb_final = accuracy_score(y_pred_lgb_final, y_test) # 0.9468666666666666

cm=classification_report(y_pred_lgb_final,y_test)


# ------------------------------ Feature Importance ------------------------------------------
rf_feature_imp = pd.DataFrame({"Value":rf_final.feature_importances_, "Feature":x_train.columns})
xgb_feature_imp = pd.DataFrame({"Value":xgboost_final.feature_importances_, "Feature":x_train.columns})
lgb_feature_imp = pd.DataFrame({"Value":lgbm_final.feature_importances_, "Feature":x_train.columns})


fig, axs = plt.subplots(1,3,figsize=(18,6))
sns.barplot(x="Value",y="Feature",data=rf_feature_imp.sort_values(by="Value",ascending=False)[0:len(x_train)],ax=axs[0])
axs[0].set_title('Feature Importance for Random Forest')
sns.barplot(x="Value",y="Feature",data=xgb_feature_imp.sort_values(by="Value",ascending=False)[0:len(x_train)],ax=axs[1])
axs[1].set_title('Feature Importance for XGBoost')
sns.barplot(x="Value",y="Feature",data=lgb_feature_imp.sort_values(by="Value",ascending=False)[0:len(x_train)],ax=axs[2])
axs[2].set_title('Feature Importance for LightGBM')
plt.tight_layout()


# ------------------------------ preparation of given data ------------------------------------------
# Now I will use the given data (tests) to predict the match results of the pokemons in the trained and optimized model.

test_data = tests.copy()
test_data.head()
test_data.shape

attack_fp = []
sp_atck_fp = []
speed_fp = []
fea_mea_fp = []
win_ra_fp = []
attack_sp = []
sp_atck_sp = []
speed_sp = []
fea_mea_sp = []
win_ra_sp = []

list_name = [attack_fp,sp_atck_fp,speed_fp,fea_mea_fp,win_ra_fp,attack_sp,sp_atck_sp,speed_sp,fea_mea_sp,win_ra_sp]

i = 0
def preparer(col,col_name):

    for val in test_data[col]:
        for id in df["ID"]:
            if id == val:
                list_name[i-1].append(df.loc[df["ID"] == id, col_name].values[0])

list_col = ["ATTACK","SP._ATK","SPEED","FEATURES_MEAN","WIN_RATE"]
for col in test_data.iloc[:,:]:
    for col_name in list_col:
        i +=1
        preparer(col,col_name)


test_pok = pd.DataFrame(data={"First_pokemon":test_data["First_pokemon"],
                             "fp_attack":attack_fp,
                             "fp_sp_attack":sp_atck_fp,
                             "fp_speed":speed_fp,
                             "fp_feature_mean":fea_mea_fp,
                             "fp_win_rate":win_ra_fp,
                             "Second_pokemon":test_data["Second_pokemon"],
                             "sp_attack": attack_sp,
                             "sp_sp_attack": sp_atck_sp,
                             "sp_speed": speed_sp,
                             "sp_feature_mean": fea_mea_sp,
                             "sp_win_rate": win_ra_sp,
                              })

test_pok.head()
#test_pok2 = test_pok.copy()
# If first pokemon win --> 0, If second pokemon win --> 1
test_pok.head()
test_pok.isnull().sum()
test_pok.fillna(0,inplace=True)

print(f"Random Forest: {round(acc_rf,5)} --> {round(acc_rf_final,5)}\n\
XGBoost:       {acc_xgb}    --> {round(acc_xgb_final,5)}\n\
LightGBM:      {acc_lgb}    --> {round(acc_lgb_final,5)}\n")

# I choose XGBClassifier predictions based on test results.
y_pred_xgb_final = xgboost_final.predict(test_pok)



# now we will combine the results and the given data
results = []
test_data.head()
for index,rslt in enumerate(y_pred_xgb_final):
    if rslt == 0:
        results.append(test_data["First_pokemon"][index])
    else:
        results.append(test_data["Second_pokemon"][index])

predictions = pd.DataFrame(data={"First_pokemon":test_data["First_pokemon"],
                             "Second_pokemon":test_data["Second_pokemon"],
                                  "Winner":results
                                  })
# top 10 results
predictions.head(10)





"""
#   -------------------------------------- FIRST POKEMON

attack_fp = []
for val in combat["First_pokemon"]:
    for id in df["ID"]:
        if id == val:
            attack_fp.append(df.loc[df["ID"]==id,"ATTACK"].values[0])
        else:
            continue

sp_atck_fp = []
for val in combat["First_pokemon"]:
    for id in df["ID"]:
        if id == val:
            sp_atck_fp.append(df.loc[df["ID"]==id,"SP._ATK"].values[0])
        else:
            continue

speed_fp = []
for val in combat["First_pokemon"]:
    for id in df["ID"]:
        if id == val:
            speed_fp.append(df.loc[df["ID"]==id,"SPEED"].values[0])
        else:
            continue

fea_mea_fp = []
for val in combat["First_pokemon"]:
    for id in df["ID"]:
        if id == val:
            fea_mea_fp.append(df.loc[df["ID"]==id,"FEATURES_MEAN"].values[0])
        else:
            continue

win_ra_fp = []
for val in combat["First_pokemon"]:
    for id in df["ID"]:
        if id == val:
            win_ra_fp.append(df.loc[df["ID"]==id,"WIN_RATE"].values[0])
        else:
            continue

#  -------------------------------------- SECOND POKEMON

attack_sp = []
for val in combat["Second_pokemon"]:
    for id in df["ID"]:
        if id == val:
            attack_sp.append(df.loc[df["ID"]==id,"ATTACK"].values[0])
        else:
            continue

sp_atck_sp = []
for val in combat["Second_pokemon"]:
    for id in df["ID"]:
        if id == val:
            sp_atck_sp.append(df.loc[df["ID"]==id,"SP._ATK"].values[0])
        else:
            continue

speed_sp = []
for val in combat["Second_pokemon"]:
    for id in df["ID"]:
        if id == val:
            speed_sp.append(df.loc[df["ID"]==id,"SPEED"].values[0])
        else:
            continue

fea_mea_sp = []
for val in combat["Second_pokemon"]:
    for id in df["ID"]:
        if id == val:
            fea_mea_sp.append(df.loc[df["ID"]==id,"FEATURES_MEAN"].values[0])
        else:
            continue

win_ra_sp = []
for val in combat["Second_pokemon"]:
    for id in df["ID"]:
        if id == val:
            win_ra_sp.append(df.loc[df["ID"]==id,"WIN_RATE"].values[0])
        else:
            continue
"""
# yukarıda ayrı ayrı olan bu döngülerin fonksiyonlaştırılmış hali bulunmaktadır.














