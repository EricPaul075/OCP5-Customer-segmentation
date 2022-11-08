# ***********************************************************************************************************************
# Liste des fonctions de ce module :
#
# Fonctions pour l'analyse exploratoire
# - polyreg : renvoie la régression linéaire
# - pair_plot : trace le nuage x,y avec possibilité de tracé de nuage partiel
# - eta_squared : calcule le rapport de corrélation entre une variable catégorielle et une variable numérique
# - df_style_fct, df_style : fonction de style pour l'affichage des dataframes de corrélation
# - sort_columns, sort_index, sort_labels: fonctions utilisées par cor_table
# - cor_table: calcule et affiche la table des corrélations de Pearson d'une liste de variables numériques 2 à 2
# - eta_table: calcule et affiche la table des rapports de corrélations entre variables catégorielles et numériques
# - welch_ttest: test de Welch (non égalité des moyennes par catégorie d'une variable continue)
# - anova: analyse et tracé graphique
#
# Fonctions pour le feature engineering
# -
#
# ***********************************************************************************************************************

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
from numpy.polynomial import polynomial as P
import scipy.stats as st
from scipy.stats import f
import seaborn as sns

# Display options
from IPython.display import display, display_html, display_png, display_svg

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 199)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

# Colorama
from colorama import Fore, Back, Style

# Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Style: DIM, NORMAL, BRIGHT, RESET_ALL

# Répertoires de sauvegarde
data_dir = '.\P5_data'


# Régression avec np.polynomial.polynomial.polyfit entre les colonnes xlabel et ylabel du DataFrame data
# Si l'argument 'full' n'est pas spécifié, renvoie les coefficients du polynôme: poly[i] coefficient du terme en X^i
# Sinon retourne 'reg' avec:
# - poly=reg[0] coefficients du polynôme: poly[i] coefficient du terme en X^i
# - list=reg[1] permet de calculer r2 = 1 - list[0][0] / (np.var(yData) * len(yData))
def polyreg(data, xlabel, ylabel, full=False, deg=1):
    xData = data[xlabel].copy(deep=True)
    yData = data[ylabel].copy(deep=True)
    if not full:
        return P.polyfit(xData, yData, deg=deg)
    else:
        return P.polyfit(xData, yData, deg=deg, full=True)


# Tracé de graphe par paire de variables numériques avec :
# - à gauche le nuage de l'ensemble des points
# - à droite, si un ou plusieurs arguments sont spécifiés, le nuage réduit avec la courbe de tendance
def pair_plot(data, pair, exclude_x=None, exclude_y=None, xmin=None, xmax=None, ymin=None, ymax=None, sharey=True):
    # Initialisation des paramètres avec les kwargs
    if exclude_x is not None:
        bx = True
    else:
        bx = False
    if exclude_y is not None:
        by = True
    else:
        by = False
    if xmin is not None:
        bxmin = True
    else:
        bxmin = False
    if xmax is not None:
        bxmax = True
    else:
        bxmax = False
    if ymin is not None:
        bymin = True
    else:
        bymin = False
    if ymax is not None:
        bymax = True
    else:
        bymax = False

    df = data[pair].apply(pd.to_numeric, axis=1)
    pts_nuage_total = len(df)
    coef_p = st.pearsonr(df[pair[0]], df[pair[1]])[0]
    # Pour éventuellement examiner la corrélation en excluant les valeurs nulles
    if not bx and not by and not bxmin and not bxmax and not bymin and not bymax:
        print("Nuage complet, Pearson=" + f"{coef_p:.2f}")
        sns.jointplot(data=df, x=pair[0], y=pair[1], kind="reg", marginal_kws=dict(bins=20, fill=True))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=sharey)
        suptitle = "Pairplot " + pair[0] + " et " + pair[1]
        fig.suptitle(suptitle)
        sns.scatterplot(ax=axes[0], data=df, x=pair[0], y=pair[1])
        title = "Nuage complet, Pearson=" + f"{coef_p:.2f}"
        axes[0].set_title(title)
        if bx:
            df = df[df[pair[0]] != exclude_x]
        if by:
            df = df[df[pair[1]] != exclude_y]
        if bxmin:
            df = df[df[pair[0]] >= xmin]
        if bxmax:
            df = df[df[pair[0]] <= xmax]
        if bymin:
            df = df[df[pair[1]] >= ymin]
        if bymax:
            df = df[df[pair[1]] <= ymax]
        pts_nuage_partiel = len(df)
        coef_p_ve = st.pearsonr(df[pair[0]], df[pair[1]])[0]
        g = sns.scatterplot(ax=axes[1], data=df, x=pair[0], y=pair[1])
        poly = polyreg(df, pair[0], pair[1])
        g.axline(xy1=(0, poly[0]), slope=poly[1], color='k', dashes=(5, 2))
        title = f"Nuage partiel ({100 * pts_nuage_partiel / pts_nuage_total:.0f}%), Pearson={coef_p_ve:.2f}"
        axes[1].set_title(title)
    plt.tight_layout()
    plt.show()
    # plt.savefig("Figure - pairplot recherche correlation.png", dpi=150)


# Calcul du rapport de corrélation entre une variable catégorielle x et une variable quantitative y
def eta_squared(x, y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x == classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj - moyenne_y) ** 2 for yj in y])
    SCE = sum([c['ni'] * (c['moyenne_classe'] - moyenne_y) ** 2 for c in classes])
    return SCE / SCT


# Fonctions de style d'affichage de dataframe avec display
# Création d'une map symétrique spécifique, source: stackoverflow
import matplotlib.colors as mcolors

discrete_palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                    'tab:brown', 'tab:pink', 'tab:grey', 'tab:olive', 'tab:cyan']

def discrete_colormap(colors=['steelblue', 'coral']):
    cmap = mpl.colors.ListedColormap(colors).with_extremes(over=colors[0], under=colors[1])
    bounds = np.linspace(0, 1, len(colors))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

def symmetrical_colormap(cmap_settings, new_name=None, shades=128):
    # get the colormap
    cmap = plt.cm.get_cmap(*cmap_settings)
    if not new_name:
        new_name = "sym_" + cmap_settings[0]  # ex: 'sym_Blues'

    # get the list of color from colormap
    colors_r = cmap(np.linspace(0, 1, shades))  # take the standard colormap # 'right-part'
    colors_l = colors_r[::-1]  # take the first list of color and flip the order # "left-part"

    # combine them and build a new colormap
    colors = np.vstack((colors_l, colors_r))
    mymap = mcolors.LinearSegmentedColormap.from_list(new_name, colors)

    return mymap


# Code type: df.style.pipe(df_style).applymap(df_style_fct)
# Seuil de mise en évidence de valeurs, à ajuster dans le Notebook
threshold = 0.5

# Ajoute une bordure rouge épaisse aux valeurs supérieures au seuil
def df_style_fct(val):
    if (abs(val) > threshold) and (val != 1):
        border = '3px solid red'
    else:
        border = '1px solid black'
    return 'border: %s' % border

# Règle le style d'affichage du dataframe
# - Nombres float avec 2 chiffres après la virgule
# - Gradient de couleurs entre les valeurs vmin et vmax
# cmap_settings = ('Blues', None)  # provide int instead of None to "discretize/bin" the colormap
cmap_settings = ('Reds', None)  # provide int instead of None to "discretize/bin" the colormap
cmap = symmetrical_colormap(cmap_settings=cmap_settings)


def df_style(styler):
    styler.format("{:.2f}")
    styler.set_table_styles([
        {'selector': 'th.col_heading',
         'props': 'text-align: center; border: 1px solid black; background-color: #4b8bbe'},
        {'selector': 'th.row_heading',
         'props': 'text-align: right; border: 1px solid black; background-color: #4b8bbe'}], overwrite=False)
    styler.background_gradient(axis=None, vmin=-1, vmax=1, cmap=cmap)
    return styler


def sort_columns(data):
    return data[sorted(data.columns.tolist())]


def sort_index(data):
    return data.reindex(sorted(data.index.tolist()))


def sort_labels(data):
    return sort_index(sort_columns(data))


def cor_table(data, num_features, thr=0):
    # Matrice de corrélation
    df_cor = data[num_features].corr(method='pearson')

    # Filtrage des lignes et colonnes dont toutes les valeurs sont < thr
    df_hit = df_cor.applymap(lambda x: 1 if abs(x)>=thr else 0)
    for idx, label in enumerate(df_cor.columns):
        if df_hit.iloc[idx].sum()<2:
            df_cor = df_cor.drop(index=[label]).drop(columns=[label])

    # Liste des paires de features dont la correlation est supérieure à threshold
    list_pairs = []
    for f1 in df_cor.columns:
        for f2 in df_cor.columns:
            if (f1!=f2) and\
                    ([f1, f2] not in list_pairs) and ([f2, f1] not in list_pairs)\
                    and (abs(df_cor.at[f1, f2])>=threshold):
                list_pairs.append([f1, f2])
    print(Fore.LIGHTBLUE_EX
          + f"  → Liste des {len(list_pairs)} paires dont la corrélation est supérieure à {threshold} :"
          + Style.RESET_ALL)
    display(list_pairs)

    # Affichage de la table avec les correlations en surbrillance
    print(Fore.LIGHTBLUE_EX + "  → Table des corrélations :" + Style.RESET_ALL)
    display(df_cor.style.pipe(df_style).applymap(df_style_fct))


# Établissement d'une table de calcul des η² par paires de variables catégorielles,
# dont la liste est dans cat_features, et numériques, dont la liste est dans num_features
# La valeur de seuil (thr) filtre les valeurs supérieures
# Possibilité d'export de la table vers MS-Excel en spécifiant le nom de fichier
def eta_table(data, cat_features, num_features, thr=0, sorted=False):
    # Matrice des corrélations
    df_eta = np.array(
        [eta_squared(data[feature], data[other_feature]) for feature in cat_features for other_feature in
         num_features])
    df_eta = pd.DataFrame(np.reshape(df_eta, (len(cat_features), len(num_features))), index=cat_features,
                          columns=num_features)

    # Test de la matrice par rapport à un seuil
    df_hit = df_eta.applymap(lambda x: x >= thr)
    print(Fore.GREEN
          + "► Recherche de corrélations potentielle entre caractéristiques catégorielles et numériques",
          ", coefficient η² >", thr, ":" + Style.RESET_ALL)

    # Liste des paires avec η² > thr et liste des exclusions par overfitting (trop de catégories)
    list_pair_eta = []
    list_pair_eta_2display = []
    cat_exclusions = []
    thr_cat_excl = (int(np.log(len(data)) ** 2 / 2.3) + 1)
    for num_feature in num_features:
        for cat_feature in df_hit.index[df_hit[num_feature] == True].tolist():
            if data[cat_feature].nunique() > thr_cat_excl:
                if cat_feature not in cat_exclusions:
                    cat_exclusions.append(cat_feature)
            elif ([cat_feature, num_feature] not in list_pair_eta):
                if df_eta.loc[cat_feature, num_feature] >= threshold:
                    list_pair_eta.append([cat_feature, num_feature])
                list_pair_eta_2display.append([cat_feature, num_feature])
    list_pair_eta.sort()

    # Affichage des exclusions par trop grand nombre de catégories
    if len(cat_exclusions) > 0:
        print(Fore.LIGHTBLUE_EX
              + "  → Recommandation d'exclusion de ces variables à grand nombre de catégories :"
              + Style.RESET_ALL)
        display(cat_exclusions)
        df_eta = sort_labels(df_eta)
        display(df_eta.loc[cat_exclusions].style.pipe(df_style).applymap(df_style_fct))

    # Affichage des paires de caractéristiques au-dessus du seuil de corrélation
    if len(list_pair_eta) > 0:
        print(Fore.LIGHTBLUE_EX
              + "  → Liste des", len(list_pair_eta), "paires de caractéristiques au-dessus du seuil :"
              + Style.RESET_ALL)
        display(list_pair_eta)
    else:
        print(Fore.LIGHTBLUE_EX
              + "  → aucune paire de caractéristiques n'est au-dessus du seuil"
              + Style.RESET_ALL)

    # Affichage de la table des corrélations
    cat_items = list(set([item[0] for item in list_pair_eta_2display]))
    cat_items = [feature for feature in cat_features if feature in cat_items]
    num_items = list(set([item[1] for item in list_pair_eta_2display]))
    num_items = [feature for feature in num_features if feature in num_items]
    df_eta = df_eta.loc[cat_items, num_items]
    if sorted:
        df_eta = sort_labels(df_eta)
    if len(df_eta) > 0:
        print(Fore.LIGHTBLUE_EX + "\n  → Table des corrélations :" + Style.RESET_ALL)
        display(df_eta.style.pipe(df_style).applymap(df_style_fct))
    else:
        print(Fore.LIGHTBLUE_EX
              + "\n  → Pas de table des corrélations pour les variables et le seuil considérés"
              + Style.RESET_ALL)


# Test de Welch, retours:
# - dof= degrés de liberté
# - t  = test de Welch
# - p  = p-value
def welch_ttest(x, y):
    dof = (x.var() / x.size + y.var() / y.size) ** 2 / (
            (x.var() / x.size) ** 2 / (x.size - 1) + (y.var() / y.size) ** 2 / (y.size - 1))
    t, p = st.ttest_ind(x, y, equal_var=False)
    return t, p, dof

# Fonction ANOVA: cacule et trace l'ANOVA pour la paire 'pair' (cat, num)
# les données sont dans le dataframe data
# alpha est le seuil des tests de p-value
# verbose=0 n'affiche que le graphique et filtre les UserWarning
# verbose=1 affiche en plus les résultats des tests, la régression linéaire
# verbose=1 affiche en plus les éventuels UserWarning
import warnings
def anova(data, pair, nb_cat=5, alpha=0.05, verbose=0):
    print(Fore.GREEN + "► ANOVA pour la paire : " + Style.RESET_ALL, pair)
    df = data[pair].copy(deep=True)

    # Filtrage des UserWarning en mode non verbose
    if verbose < 2:
        warnings.filterwarnings(action="ignore", category=UserWarning)

    # Filtrage des catégories qui contiennent moins de 'Nb_prod_per_cat_min' lignes (min=3)
    df_cat = df.groupby(pair[0], as_index=False).agg(means=(pair[1], "mean"), size=(pair[0], "size")).sort_values(
        by='means', ascending=False).reset_index(drop=True)
    Nb_prod_per_cat_min = 3
    list_cat = df_cat.loc[df_cat['size'] >= Nb_prod_per_cat_min, pair[0]].tolist()
    df.drop(index=df.loc[~df[pair[0]].isin(list_cat), :].index, inplace=True)
    df_cat.drop(index=df_cat.loc[~df_cat[pair[0]].isin(list_cat), :].index, inplace=True)
    df_cat.reset_index(drop=True, inplace=True)

    # Filtrage sur les nb_cat pour lesquelles la moyenne des valeurs numériques est la plus élevée
    df_cat = df_cat.head(nb_cat)
    list_cat = df_cat[pair[0]].head(nb_cat).values.tolist()
    nb_cat = min(nb_cat, len(list_cat))
    df.drop(index=df.loc[~df[pair[0]].isin(list_cat), :].index, inplace=True)
    df[pair[0]] = pd.Categorical(df[pair[0]], categories=list_cat, ordered=True)

    # Calcul du rapport de corrélation
    eta_sqr = eta_squared(df[pair[0]], df[pair[1]])
    if verbose > 0:
        print("  → Rapport de corrélation pour les k=", nb_cat, "catégories du graphique et n=",
              df.shape[0], "données : η²=" + f"{eta_sqr:.2f}")

    # Remplacement des catégories par une valeur numérique
    df['cat'] = df[pair[0]].copy()
    df['cat'] = df['cat'].astype('object').astype("category")
    df['cat'].replace(df['cat'].cat.categories, [i for i in range(0, len(df['cat'].cat.categories))], inplace=True)
    df['cat'] = df['cat'].astype("int")

    # Tests sur les variables
    # Test sur la condition de loi normale
    tn = True
    list_shapiro_neg = {'category': [], 'statistic': [], 'p-value': []}
    for cat in range(nb_cat):
        ts = st.shapiro(df.loc[df['cat'] == cat, pair[1]].values)
        tn = tn and (ts.pvalue <= alpha)
        if ts.pvalue > alpha:
            list_shapiro_neg['category'].append(cat)
            list_shapiro_neg['statistic'].append(ts.statistic)
            list_shapiro_neg['p-value'].append(ts.pvalue)
    if verbose > 0:
        if tn:
            print("  → Test de normalité de Shapiro positif pour toutes les catégories")
        else:
            print("  → Test de normalité de Shapiro négatif sur certaines catégories")
            if verbose:
                print("  \tCatégories concernées et résultat du test :")
                display(pd.DataFrame.from_dict(list_shapiro_neg))

    # Test d'homoscédasticité (écarts types égaux entre les catégories)
    gb = df.groupby(pair[0])[pair[1]]
    stat, p_bartlett = st.bartlett(*[gb.get_group(x).values for x in gb.groups.keys()])
    if verbose > 0:
        if p_bartlett <= alpha:
            print(f"  → Test d'homoscédasticité de Bartlett négatif : p-value={p_bartlett:.2e}")
            if verbose > 1:
                std = pd.DataFrame(data=[gb.get_group(x).values.std() for x in gb.groups.keys()],
                                   columns=['std'], index=gb.groups.keys())
                print("\n\tEcarts types:")
                display(std)
        else:
            print("  → Test d'homoscédasticité de Bartlett positif (écarts types égaux entre les catégories):",
                  f"p_bartlett={p_bartlett:.2e}")

    # Test de Welch, dans le cas où le test d'homoscédasticité est négatif
    tw = True
    # Table de groupe des catégories en fonction du résultat du test
    dgr = pd.DataFrame(data=np.arange(len(list_cat)), index=[list_cat], columns=['group'])
    for i in range(len(list_cat) - 1):
        for j in range(i + 1, len(list_cat)):
            t, p, dof = welch_ttest(gb.get_group(list_cat[i]).values, gb.get_group(list_cat[j]).values)
            # Si le test est négatif, les 2 catégories appartiennent au même groupe
            if p > alpha:
                tw = False
                gr = dgr.loc[list_cat[i]]['group']
                dgr.at[list_cat[j], 'group'] = gr
    # Valeurs de l'ordonnée pour le grouper les catégories ayant des moyennes non dissemblables
    rows = [-0.5]
    for i in range(1, len(list_cat)):
        if dgr['group'].values[i]!=dgr['group'].values[i-1]:
            rows.append(i-0.5)
    rows.append(len(list_cat)-0.5)
    # Affichage du résultat du test de Welch
    if verbose > 0:
        if tw:
            print("  → Test de Welch (non égalité des moyennes) positif pour toutes les catégories")
        else:
            print("  → Test de Welch négatif entre les catégories de même groupe sur le graphique")

    # Test statistique de Fisher
    dfn = nb_cat - 1
    dfd = df.shape[0] - nb_cat
    F_crit = f.ppf(1 - alpha, dfn, dfd)
    F_stat, p = st.f_oneway(df['cat'], df[pair[1]])
    sign_F = ">" if F_stat > F_crit else "<"
    sign_p = ">" if p > alpha else "<"
    if (sign_F == ">") and (sign_p == "<"):
        res_test = "positif"
    else:
        res_test = "négatif"
    if verbose > 0:
        print(f"  → Résultat {res_test} du test de Fisher : F={F_stat:.2f} {sign_F} {F_crit:.2f}",
              f" , et p-value={p:.2e} {sign_p} {alpha:0.2f}")
        print(Fore.LIGHTBLACK_EX + " Rappel des hypothèses relatives au test :")
        print("  - H0 : les moyennes par catégories sont égales entre elles (les variables sont indépendantes)")
        print("  - H1 : la moyenne d'au moins une catégorie diffère des autres (les variables sont corrélées)" +
              Style.RESET_ALL)

    # Définition des dimensions du graphique global
    fig_h = nb_cat if nb_cat < 6 else int((5 * nb_cat + 40) / 15)

    # Propriétés graphiques
    medianprops = {'color': "black"}
    meanprops = {'marker': 'o', 'markeredgecolor': 'black', 'markerfacecolor': 'firebrick'}

    rcParams['figure.figsize'] = 10, fig_h
    ax = sns.boxplot(x=pair[1], y=pair[0], data=df, showfliers=False,
                     medianprops=medianprops, showmeans=True, meanprops=meanprops)
    xmin, xmax = ax.get_xlim()

    # Tracé des lignes reliant les valeurs moyennes de chaque catégorie
    plt.plot(df_cat.means.values, df_cat.index.values, linestyle='--', c='#000000')

    # Bloc de séparation graphique des groupes de moyennes non différenciées (test de Welch négatif)
    if not tw and len(rows)>1:
        for i in range(len(rows)-1):
            plt.fill_between([xmin, xmax], [rows[i], rows[i]], [rows[i+1], rows[i+1]], alpha=0.2)

    # Régression linéaire sur les valeurs moyennes
    reg = P.polyfit(df_cat.means.values, df_cat.index.values, deg=1, full=True)
    yPredict = P.polyval(df_cat.means.values, reg[0])

    if verbose > 0:
        if nb_cat > 2:
            coef_cor = 1 - reg[1][0][0] / (np.var(df_cat.index.values) * len(df_cat.index.values))
        else:
            coef_cor = 1
        a = -1 / reg[0][1]
        mu = -reg[0][0] / reg[0][1] - a * (df_cat.shape[0] - 1)
        sign = '+' if a >= 0 else '-'
        print(f"\n  → Moyenne catégorielle : '{pair[1]}' = {mu:.2f}  {sign} {abs(a):.2f} * '{pair[0]}', avec :",
              f"'{df_cat[pair[0]][df_cat.shape[0] - 1]}'= 0 , …, '{df_cat[pair[0]][0]}'= {df_cat.shape[0] - 1}")
        print(f"  → Coefficient de corrélation r² ={coef_cor:.2f}")

    # Tracé de la droite de régression linéaire
    plt.plot(df_cat.means.values, yPredict, linewidth=2, linestyle='-', c='#FF0000')

    plt.ylim(top=-1, bottom=nb_cat)
    plt.tight_layout()
    plt.show()

    # Retour à la gestion des warnings par défaut
    if verbose > 1:
        warnings.filterwarnings(action="default", category=UserWarning)


# *********************************************************************************************************************
# Fonctions de transformation des variables
# Paramètre de transformation avec StandardScaler
ssc_tr_param = {'fname': [], 'mean': [], 'std': []}

def set_param_ssc(feature_name, feature_mean, feature_std):
    if feature_name in ssc_tr_param.get('fname', []):
        index = ssc_tr_param.get('fname').index(feature_name)
        ssc_tr_param['mean'][index] = feature_mean
        ssc_tr_param['std'][index] = feature_std
    else:
        ssc_tr_param['fname'].append(feature_name)
        ssc_tr_param['mean'].append(feature_mean)
        ssc_tr_param['std'].append(feature_std)

def get_param_ssc(feature_name):
    index = ssc_tr_param.get('fname').index(feature_name)
    return ssc_tr_param.get('mean')[index], ssc_tr_param.get('std')[index]

from json import dump
def save_param_ssc(dic=ssc_tr_param, dic_file=data_dir + "\\" + 'ssc_tr_param'):
    file = open(dic_file + '.json', 'w')
    dump(dic, file)
    file.close()

from json import loads
def read_param_ssc(dic_file=data_dir + "\\" + 'ssc_tr_param'):
    file = open(dic_file + '.json', 'r')
    json_string = file.read()
    file.close()
    return loads(json_string)


# *********************************************************************************************************************
# Fonctions pour ACP
#
# * display_circle(ax, pcs, n_comp, pca, plan, c_labels=None, c_label_rotation=0, c_lims=None, filter=None)
#   affiche le cercle des corrélations du plan=[i,j] avec i et j dans [1, n_comp] et i≠j
#   en fixant optionnellement c_lims = xmin, xmax, ymin, ymax
#   en filtrant optionnellement les vecteurs de longueur inférieurs à filter
#
#
from matplotlib.collections import LineCollection

def display_circle(ax, pcs, n_comp, pca, plan, c_labels=None, c_label_rotation=0,
                   c_lims=None, filter=None):
    d1 = plan[0] - 1
    d2 = plan[1] - 1
    if (d1 not in range(0, n_comp)) or (d1 not in range(0, n_comp)):
        print("Numéro d'axe d'inertie or limite")
        return

    if filter is not None:
        b = (pcs[d1, :] * pcs[d1, :] + pcs[d2, :] * pcs[d2, :]) > filter**2
        pcs[d1, :] = pcs[d1, :] * b
        pcs[d2, :] = pcs[d2, :] * b

    # détermination des limites du graphique
    if c_lims is not None:
        xmin, xmax, ymin, ymax = c_lims
    elif pcs.shape[1] > 30:
        xmin, xmax, ymin, ymax = -1, 1, -1, 1
    else:
        e = 0.0
        xmin, xmax, ymin, ymax = min(pcs[d1, :])-e, max(pcs[d1, :])+e, min(pcs[d2, :])-e, max(pcs[d2, :])+e
        xmin, xmax, ymin, ymax = max(min(xmin,0),-1), min(max(0,xmax),1), max(min(ymin,0),-1), min(max(0,ymax),1)


        # affichage des flèches
        # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
    if pcs.shape[1] < 30:
        ax.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1, :], pcs[d2, :],
                   angles='xy', scale_units='xy', scale=1, color="grey")
        # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
    else:
        lines = [[[0, 0], [x, y]] for x, y in pcs[[d1, d2]].T]
        ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

    if c_labels is not None:
        for i, (x, y) in enumerate(pcs[[d1, d2]].T):
            #if (x != 0 and y != 0) and x >= xmin and x <= xmax and y >= ymin and y <= ymax:
            if not(x==0 and y==0):
                ax.text(x, y, c_labels[i],
                        fontsize='10', ha='center', va='center', rotation=c_label_rotation, alpha=1)

    # affichage du cercle
    circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
    ax.set_aspect(1)
    ax.add_artist(circle)

    # définition des limites du graphique
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # affichage des lignes horizontales et verticales
    ax.plot([-1, 1], [0, 0], color='grey', ls='--')
    ax.plot([0, 0], [-1, 1], color='grey', ls='--')

    # nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel('F{} ({}%)'.format(d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)))
    ax.set_ylabel('F{} ({}%)'.format(d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)))

    ax.set_title("Cercle des corrélations (F{} et F{})".format(d1 + 1, d2 + 1))



def display_factorial_plan(ax, X_projected, n_comp, pca, plan, scale=None,
                           p_labels=None, alpha=1, illustrative_var=None,
                           color=None, cmap='YlOrRd', centroids=None, cmarker='X'):
    #out = None
    d1 = plan[0]-1
    d2 = plan[1]-1
    if (d1 not in range(0, n_comp)) or (d1 not in range(0, n_comp)):
        print("Numéro d'axe d'inertie hors limite")
        return

    # affichage des points
    if illustrative_var is None:
        if color is None:
            out = ax.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
        else:
            out = ax.scatter(X_projected[:, d1], X_projected[:, d2], c=color, cmap=cmap, alpha=alpha)
    else:
        illustrative_var = np.array(illustrative_var)
        for value in np.unique(illustrative_var):
            selected = np.where(illustrative_var == value)
            if color is None:
                out = ax.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
            else:
                out = ax.scatter(X_projected[selected, d1], X_projected[selected, d2],
                                c=color, cmap=cmap, alpha=alpha, label=value)
        ax.legend()

    # affichage des labels des points
    if p_labels is not None:
        for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
            ax.text(x, y, p_labels[i], fontsize='10', ha='center', va='center')

    # affichage des centroïdes
    if centroids is not None:
        ax.scatter(centroids[:, d1], centroids[:, d2], c='black', marker=cmarker, alpha=1)

    # détermination des limites du graphique
    boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
    ax.set_xlim(-boundary, boundary)
    ax.set_ylim(-boundary, boundary)

    # détermination de l'échelle du graphique
    if scale is None:
        scalex = 'linear'
        scaley = 'linear'
    else:
        scalex, scaley = scale
    ax.set_xscale(scalex)
    ax.set_yscale(scaley)

    # affichage des lignes horizontales et verticales
    ax.plot([-100, 100], [0, 0], color='grey', ls='--')
    ax.plot([0, 0], [-100, 100], color='grey', ls='--')

    # nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f"F{d1+1} ({100*pca.explained_variance_ratio_[d1]:.1f}%) - scale: '{str(scalex)}'")
    ax.set_ylabel(f"F{d2+1} ({100*pca.explained_variance_ratio_[d2]:.1f}%) - scale: '{str(scaley)}'")

    ax.set_title(f"Projection des individus (sur F{d1+1} et F{d2+1})")

    return out


def project_plot(ax, X_projected, n_comp, pca, plan, scale=None, alpha=1,
                 illustrative_var=None, cmap='YlOrRd', color=None,
                 centroids=None, cmarker='X', p_labels=None, save=None):

    out = display_factorial_plan(ax, X_projected, n_comp, pca, plan, scale=scale,
                                 p_labels=p_labels, alpha=alpha,
                                 illustrative_var=illustrative_var,
                                 color=color, cmap=cmap,
                                 centroids=centroids, cmarker=cmarker)

    if save is not None:
        plt.savefig(save, dpi=300)

    return out

def circle_plot(ax, pcs, n_comp, pca, plan, c_labels=None,c_label_rotation=0,
                c_lims=None, filter=None, save=None):

    display_circle(ax, pcs, n_comp, pca, plan,
                   c_labels=c_labels, c_label_rotation=c_label_rotation,
                   c_lims=c_lims, filter=filter)

    if save is not None:
        plt.savefig(save, dpi=150)


def projetNcircle_plot(pcs, X_projected, n_comp, pca, plan, scale=None, alpha=1,
                       illustrative_var=None, color=None, cmap='YlOrRd', cnames=None,
                       centroids=None, cmarker='X', p_labels=None, c_labels=None,
                       c_label_rotation=0, c_lims=None, filter=None, save=None):

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Cercle des corrélation et projection dans le plan factoriel F{}-F{}".format(plan[0], plan[1]),
                 fontweight='bold')

    circle_plot(axs[0], pcs, n_comp, pca, plan,
                c_labels=c_labels, c_label_rotation=c_label_rotation, c_lims=c_lims, filter=filter)

    out = project_plot(axs[1], X_projected, n_comp, pca, plan, scale=scale, alpha=alpha,
                      illustrative_var=illustrative_var, color=color, cmap=cmap,
                      centroids=centroids, cmarker=cmarker, p_labels=p_labels, save=None)

    if (color is not None) and (cnames is not None):
        cbar = fig.colorbar(out, ax=axs[1])
        cbar.set_ticks(np.unique(color))
        cbar.set_ticklabels(cnames)
    elif color is not None:
        fig.colorbar(out, ax=axs[1])

    plt.tight_layout()
    plt.show()
    if save is not None:
        print("\t→ Sauvegarde du tracé dans :", save)
        plt.savefig(save, dpi=300)


def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_ * 100
    plt.bar(np.arange(len(scree)) + 1, scree, label='Inertie par composante')
    plt.plot(np.arange(len(scree)) + 1, scree.cumsum(), c="red", marker='o', label='Cumul inertie')
    kaiser_crit = 100 / np.shape(pca.components_)[1]
    plt.plot([1, len(scree)], [kaiser_crit, kaiser_crit], c='green', label='Critère de Kaiser')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.legend(loc='center right')
    plt.title("Éboulis des valeurs propres")
    plt.tight_layout()
    plt.show(block=False)

#***********************************************************************************************************************

# Formate le temps écoulé entre 2 timeit.default_timer()
import timeit
def elapsed_format(elapsed):
    duration = datetime.utcfromtimestamp(elapsed)
    if elapsed >= 3600:
        return f"{duration.strftime('%H:%M:%S')}"
    elif elapsed >= 60:
        return f"{duration.strftime('%M:%S')}"
    elif elapsed >=1:
        return f"{duration.strftime('%S.%f')[:-3]}s"
    else:
        return f"{duration.strftime('%f')[:-3]}ms"


# Détermination du nombre de clusters optimal avec K-Means
from sklearn.cluster import KMeans
from sklearn import metrics
import timeit
from datetime import timedelta, datetime
def kmeans_metric_plot(X, ks=np.arange(2, 10), eval=['silhouette', 'calinski_harabasz', 'davies_bouldin']):

    inertia = []

    if 'silhouette' in eval:
        silhouette = []
        elapsed_sil = 0

    if 'calinski_harabasz' in eval:
        calinski_harabasz = []
        elapsed_ch = 0

    if 'davies_bouldin' in eval:
        davies_bouldin = []
        elapsed_db = 0

    for k in ks:
        model = KMeans(n_clusters=k).fit(X)

        inertia.append(model.inertia_)

        if 'silhouette' in eval:
            start_time = timeit.default_timer()
            silhouette.append(metrics.silhouette_score(X, model.labels_, metric='euclidean'))
            elapsed_sil += timeit.default_timer() - start_time

        if 'calinski_harabasz' in eval:
            start_time = timeit.default_timer()
            calinski_harabasz.append(metrics.calinski_harabasz_score(X, model.labels_))
            elapsed_ch += timeit.default_timer() - start_time

        if 'davies_bouldin' in eval:
            start_time = timeit.default_timer()
            davies_bouldin.append(metrics.davies_bouldin_score(X, model.labels_))
            elapsed_db += timeit.default_timer() - start_time

    if 'silhouette' in eval: elapsed_sil = elapsed_sil / len(silhouette)
    if 'calinski_harabasz' in eval: elapsed_ch = elapsed_ch / len(calinski_harabasz)
    if 'davies_bouldin' in eval: elapsed_db = elapsed_db / len(davies_bouldin)

    # Affichage graphique du score en fonction du nombre de clusters
    nrows = int((len(eval)+1) / 2) + (len(eval)+1) % 2
    ncols = 2 if len(eval)>0 else 1
    plt.figure(figsize=(6*ncols, 4*nrows))

    plt.subplot(nrows, ncols, 1)
    plt.plot(ks, inertia)
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.title("Inertie selon le nombre de clusters")

    i = 2
    if 'silhouette' in eval:
        plt.subplot(nrows, ncols, i)
        plt.plot(ks, silhouette)
        plt.xlabel('Nombre de clusters')
        plt.ylabel('Silhouette')
        plt.title(f"Silhouette (calcul={elapsed_format(elapsed_sil)})")
        i += 1

    if 'calinski_harabasz' in eval:
        plt.subplot(nrows, ncols, i)
        plt.plot(ks, calinski_harabasz)
        plt.xlabel('Nombre de clusters')
        plt.ylabel('Calinski-Harabasz')
        plt.title(f"Calinski-Harabasz (calcul={elapsed_format(elapsed_ch)})")
        i += 1

    if 'davies_bouldin' in eval:
        plt.subplot(nrows, ncols, i)
        plt.plot(ks, davies_bouldin)
        plt.xlabel('Nombre de clusters')
        plt.ylabel('Davies-Bouldin')
        plt.title(f"Davies-Bouldin (calcul={elapsed_format(elapsed_db)})")

    plt.tight_layout()
    plt.show()


#***********************************************************************************************************************
# Création de la matrice de linkage, notamment pour dessiner le dendrogramme
# Issu du modèle AgglomerativeClustering de sklearn.cluster avec compute_distances=True
def linkage_slAC(model):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return linkage_matrix

#***********************************************************************************************************************
#
def get_clusters_info(data, features, clusters_col, cluster_centers=None):
    df = data[clusters_col].value_counts().sort_index(ignore_index=True)
    df = pd.DataFrame(data=df.values, columns=['size']).rename_axis(index=['cluster'])
    if -1 in df.index: df.drop(index=[-1], inplace=True)

    if cluster_centers is not None:
        centroids = [f"centroid_{feature}" for feature in features]
        df[centroids] = cluster_centers

    mins = [f"min_{feature}" for feature in features]
    maxs = [f"max_{feature}" for feature in features]
    df[mins] = [data.loc[data[clusters_col]==c, features].min(axis=0) for c in df.index]
    df[maxs] = [data.loc[data[clusters_col]==c, features].max(axis=0) for c in df.index]

    columns = ['size']
    for feature in features:
        if cluster_centers is not None:
            columns.extend([f'min_{feature}', f'centroid_{feature}', f'max_{feature}'])
        else:
            columns.extend([f'min_{feature}', f'max_{feature}'])
    df = df[columns]

    for col in df.columns.tolist():
        if len(str(col).split('_', 2))>1:
            mean, std = get_param_ssc(str(col).split('_', 2)[1])
            df[col] = df[col] * std + mean

    return df

def get_centroids(data, features, clusters):
    data['clusters'] = clusters
    centroids = [data.loc[data['clusters'] == k, features].mean(axis=0) for k in np.unique(clusters)]
    data.drop(columns=['clusters'], inplace=True)
    return np.array(centroids)


# Renvoie la matrice sauvegardée dans le fichier s'il existe
# Renvoie True si le fichier existe
def load_matrix(filename):
    try:
        matrix = np.load(filename)
        #matrix.close()
        return True, matrix
    except:
        return False, None