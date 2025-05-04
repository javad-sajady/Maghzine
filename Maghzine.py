import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import itertools
import warnings
warnings.filterwarnings('ignore')

# Function to reverse Persian text for plotting
def reverse(a):
    l = ""
    A = a.split(" ")
    for i in A:
        s = ""
        for j in i:
            s = j + s
        l = s + l + " "
    return l

# Custom correlation function excluding NaNs
def Corr(A, B):
    def non_finder(A):
        l_ = np.ones(A.shape[0])
        t = 0
        for i in A:
            if not np.isfinite(i):
                l_[t] = 0
            t += 1
        return l_
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    L_1 = non_finder(A)
    L_2 = non_finder(B)
    s1, s2 = [], []
    for i in range(L_1.size):
        if L_1[i] == 1 and L_2[i] == 1:
            s1.append(A[i])
            s2.append(B[i])
    s1, s2 = np.array(s1), np.array(s2)
    if s1.size == 0 or s2.size == 0:
        return 0
    k = np.average(s1 * s2) - np.average(s1) * np.average(s2)
    m = np.sqrt(np.var(s1) * np.var(s2))
    return k / m if m != 0 else 0

# Sequence-based scoring
def seq_cal(arr, p):
    array = arr.copy()
    array.append(0)
    A, t, s = 0, 0, 0
    for a in array:
        if a == 1:
            t += 1
            s += 1
        else:
            if t > 0:
                A += t * np.power(p, t)
                t = 0
    return A / s / np.power(p, s) if s > 0 else 0

# Load and preprocess grade data
def preprocess_grades():
    Grade = pd.read_excel("5g.xlsx")
    Grade.rename(columns={'کد دانش آموزی': "id", 'درس': "courses"}, inplace=True)
    Grade['value'] /= 0.125
    
    # Group extracurricular subjects
    arts = ['سفال', 'عکاسی', 'لگو', 'نمایش']
    sports = ['بسکتبال', 'والیبال', 'ژیمناستیك', 'هندبال']
    for i in arts:
        Grade.loc[Grade['courses'] == i, 'courses'] = "هنری"
    for j in sports:
        Grade.loc[Grade['courses'] == j, 'courses'] = "ورزشی"
    
    # Train-test split
    np.random.seed(1234)
    test_ratio = 0.2
    test_len = int(len(set(Grade['id'])) * test_ratio)
    Ideas = list(set(Grade['id']))
    test_num = []
    while len(test_num) < test_len:
        a = Ideas[np.random.randint(0, len(Ideas))]
        if a not in test_num:
            test_num.append(a)
    train_Grade = Grade[~Grade["id"].isin(test_num)]
    
    # Normalize grades
    def normalizer(train_Grade, Grade):
        A = set(Grade['courses'])
        B_ = {}
        for i in A:
            k = np.array(train_Grade[train_Grade['courses'] == i]["value"])
            B_[i] = (k.mean(), np.var(k) if np.var(k) > 0 else 1)
        courses = set(Grade['courses'])
        Grade_ = {}
        for i in set(Grade['id']):
            l_ = []
            for j in courses:
                k = Grade[(Grade['id'] == i) & (Grade['courses'] == j)]
                if k.size > 0:
                    v = np.double(k["value"])
                    a, b = B_[j]
                    l_.append((v - a) / b)
                else:
                    l_.append(np.nan)
            Grade_[i] = l_
        k = pd.DataFrame(Grade_, index=courses).T
        k['هنر ۱'] = k['هنری'].apply(lambda x: np.nanmax(x) if not np.all(np.isnan(x)) else np.nan)
        k['هنر ۲'] = k['هنری'].apply(lambda x: np.nanmin(x) if not np.all(np.isnan(x)) else np.nan)
        k['ورزش ۱'] = k['ورزشی'].apply(lambda x: np.nanmax(x) if not np.all(np.isnan(x)) else np.nan)
        k['ورزش ۲'] = k['ورزشی'].apply(lambda x: np.nanmin(x) if not np.all(np.isnan(x)) else np.nan)
        k.drop(['ورزشی', 'هنری'], axis=1, inplace=True)
        return pd.DataFrame(np.array(k, dtype='float'), index=k.index, columns=k.columns)
    
    Grade_normalized = normalizer(train_Grade, Grade)
    Grade_full = normalizer(Grade, Grade)
    
    # Rank grades
    subject_list = {}
    for subject in Grade_normalized.columns:
        V = list(set(Grade_normalized[subject]))
        V = [v for v in V if not np.isnan(v)]
        V.sort(reverse=True)
        new_grade = [V.index(a) + 1 if not np.isnan(a) else np.nan for a in Grade_normalized[subject]]
        subject_list[subject] = new_grade
    ranking_grade = pd.DataFrame(subject_list, index=Grade_normalized.index)
    
    # Visualization: Normalization difference
    plt.figure(figsize=(15, 15), dpi=80)
    v = np.array(Grade_normalized, dtype='float') - np.array(Grade_full, dtype='float')
    sns.heatmap(v, yticklabels=Grade_normalized.index, xticklabels=[reverse(i) for i in Grade_normalized.columns],
                annot=True, linewidths=0.5, cmap="YlGnBu").set(title="Normalization: Train vs Full Dataset")
    plt.savefig('normalization_diff.png')
    plt.close()
    
    # Grade correlations
    plt.figure(figsize=(13, 10), dpi=80)
    sns.heatmap(Grade_normalized.corr(), annot=True, linewidths=0.5, cmap="YlGnBu",
                yticklabels=[reverse(i) for i in Grade_normalized.columns],
                xticklabels=[reverse(i) for i in Grade_normalized.columns])
    plt.savefig('grade_correlations.png')
    plt.close()
    
    return Grade_normalized, ranking_grade, train_Grade, test_num

# Load and preprocess exam data
def preprocess_exam():
    Exam = pd.read_excel("5.xlsx")
    Exam.rename(columns={'امتیاز': "score", 'سطح': "lev", "نام کاربری": "id", "پاسخ های درست": "NT",
                         "پاسخ های غلط": "NF", "رشته درست و غلط": "seq", 'زمان واکنش(میلی ثانیه)': "seqlen",
                         "نام بازی": "game"}, inplace=True)
    Exam.drop(["زمان", "کاربر", "جزئیات", "پایان یافته", "بیشتر", "مدت(ثانیه)"], axis=1, inplace=True)
    
    def seqTsecond(v):
        if isinstance(v, str):
            return sum(int(i) for i in v.split(","))
        return 0
    
    Exam["len"] = Exam["seqlen"].apply(seqTsecond)
    Exam = Exam[Exam.len > 0]
    Exam["Nscore"] = (Exam["NT"] - Exam["NF"]) / (Exam["len"] / 1000)
    
    # Process sequences
    def array_creator(array):
        array = np.array(array.split(","), dtype=int)
        if array.size > 1:
            a, b, c = np.quantile(array, (0.25, 0.5, 0.75))
            min_ = b - (b - a) * 2.5
            max_ = b + (c - b) * 2.5
            return np.array([1 if min_ <= v <= max_ else 0 for v in array])
        return np.array([1])
    
    def seq_creator(array):
        return np.array([1 if a == "T" else 0 for a in array])
    
    def mean_finder(array):
        return np.array(array.split(","), dtype=int)
    
    Exam['array'] = Exam['seqlen'].apply(array_creator)
    Exam['TFarray'] = Exam['seq'].apply(seq_creator)
    Exam['seq_array'] = Exam['seqlen'].apply(mean_finder)
    Exam['AT'] = (Exam['array'] * Exam['TFarray']).apply(lambda x: x.sum())
    Exam['AF'] = (Exam['array'] * (1 - Exam['TFarray'])).apply(lambda x: x.sum())
    Exam['Aseq'] = (Exam['array'] * Exam['seq_array']).apply(lambda x: x.sum())
    
    return Exam

# Compute game performance metrics
def compute_metrics(Exam, test_num):
    games = set(Exam['game'])
    ids = set(Exam['id'])
    
    # B1 and B2 metrics
    result_1, result_2 = {}, {}
    for i in ids:
        l_, m_ = [], []
        for j in games:
            A = Exam[(Exam.id == i) & (Exam['game'] == j)]
            if A.shape[0] > 0:
                a = ((A["NT"]).sum() - (A["NF"]).sum()) / (A["len"].sum() / 1000 + 1e-6)
                b = A["Nscore"].sum() / A.shape[0] if A.shape[0] > 0 else 0
                l_.append(a)
                m_.append(b)
            else:
                l_.append(np.nan)
                m_.append(np.nan)
        result_1[i], result_2[i] = l_, m_
    
    Exam_1 = pd.DataFrame(result_1, index=games).T
    Exam_2 = pd.DataFrame(result_2, index=games).T
    
    # Sequence-based metrics
    dic = []
    for game in games:
        for id in ids:
            for lev in set(Exam[Exam['game'] == game]['lev']):
                if Exam[(Exam['id'] == id) & (Exam['game'] == game) & (Exam['lev'] == lev)].size > 0:
                    a = Exam[(Exam['id'] == id) & (Exam['game'] == game) & (Exam['lev'] == lev)]['NT'].sum()
                    b = Exam[(Exam['id'] == id) & (Exam['game'] == game) & (Exam['lev'] == lev)]['NF'].sum()
                    p = (a + b) / (a + 1e-6)
                    array, t = 0, 0
                    for k in Exam[(Exam['id'] == id) & (Exam['game'] == game) & (Exam['lev'] == lev)]['TFarray']:
                        v = seq_cal(list(k), p)
                        array += v
                        t += 1
                    array = array / t if t > 0 else np.nan
                    avg_time, v = 0, 0
                    for k in Exam[(Exam['id'] == id) & (Exam['game'] == game) & (Exam['lev'] == lev)]['seq_array']:
                        for d in k:
                            avg_time += int(d)
                            v += 1
                    avg_time = avg_time / v if v > 0 else np.nan
                    dic.append((game, id, lev, 1/p, array, avg_time, a + b))
                else:
                    dic.append((game, id, lev, np.nan, np.nan, np.nan, np.nan))
    
    new_Exam = pd.DataFrame(dic, columns=["game", "id", "lev", "prob", "value", "time", "num"])
    
    # Game rankings
    ranking_dic = {}
    for game in games:
        list_ = []
        for id in ids:
            if new_Exam[(new_Exam["game"] == game) & (new_Exam["id"] == id)].size > 0:
                a = new_Exam[(new_Exam["game"] == game) & (new_Exam["id"] == id)]["value"].mean()
                b = new_Exam[(new_Exam["game"] == game) & (new_Exam["id"] == id)]["num"].mean()
                list_.append((id, a if not np.isnan(a) else 0, b if not np.isnan(b) else 0))
        list_.sort(key=lambda x: x[1] * 1e10 + x[2], reverse=True)
        final_list = {}
        t, m = 1, (0, 0)
        for a in list_:
            curr = (a[1], a[2])
            if curr != m:
                m = curr
                t += 1
            final_list[a[0]] = t
        A = sorted(list(ids))
        new_list = [final_list[a] if a in final_list else t + 1 for a in A]
        ranking_dic[game] = new_list
    ranking = pd.DataFrame(ranking_dic, index=A)
    
    # Visualization: Game ranking correlations
    plt.figure(figsize=(10, 8), dpi=80)
    sns.heatmap(ranking.corr(), annot=True, linewidths=0.5, cmap="YlGnBu",
                xticklabels=[reverse(g) for g in games], yticklabels=[reverse(g) for g in games])
    plt.savefig('game_ranking_correlations.png')
    plt.close()
    
    return Exam_1, Exam_2, new_Exam, ranking

# Cognitive factor analysis
def cognitive_factors(ranking):
    vigilance_list = ['گالری عکس', 'مو نزنه', 'حافظه عسلی', 'گروه سرود', 'مسئول کنترل']
    processing_speed_list = ['مسئول کنترل', 'ابرهای بارانی', 'بزرگراه', 'خوشمزه یاب', 'گمشده در دریا']
    working_memory_list = ['مسئول کنترل', 'ابرهای بارانی', 'حافظه عسلی', 'گروه سرود', 'گالری عکس']
    divided_attention_list = ['ابرهای بارانی']
    alternation_list = ['گروه سرود', 'خوشمزه یاب', 'مو نزنه', 'گالری عکس']
    response_inhibition_list = ['گمشده در دریا', 'بزرگراه']
    selective_attention_list = ['خوشمزه یاب', 'مو نزنه', 'گمشده در دریا', 'بزرگراه']
    sustained_list = ['حافظه عسلی', 'گروه سرود', 'مو نزنه', 'گالری عکس', 'مسئول کنترل']
    focused_and_vigilance_list = ['مسئول کنترل']
    
    list_of_list = [vigilance_list, processing_speed_list, working_memory_list, divided_attention_list,
                    alternation_list, response_inhibition_list, selective_attention_list, sustained_list,
                    focused_and_vigilance_list]
    rank_name = ["vigilance", "processing_speed", "working_memory", "divided_attention", "alternation",
                 "response_inhibition", "selective_attention", "sustained", "focused_and_vigilance"]
    
    data_set = {}
    for rank, name in zip(list_of_list, rank_name):
        b = np.array(ranking[rank])
        b = np.where(np.isnan(b), np.nanmean(b, axis=0), b)
        pca = PCA(n_components=1)
        pca.fit(b)
        data_set[name] = pca.transform(b).reshape(-1)
    
    Exam_Data = pd.DataFrame(data_set, index=ranking.index)
    
    # Visualization: Cognitive factor correlations
    plt.figure(figsize=(10, 8), dpi=80)
    sns.heatmap(Exam_Data.corr(), annot=True, linewidths=0.5, cmap="YlGnBu")
    plt.savefig('cognitive_factor_correlations.png')
    plt.close()
    
    return Exam_Data

# Factor analysis
def factor_analysis(Grade_normalized, Exam_1):
    final_data = pd.concat([Grade_normalized, Exam_1], axis=1)
    final_data = final_data.dropna()
    
    pca_1 = PCA(n_components=8)
    pca_1.fit(final_data[Grade_normalized.columns])
    A_1 = pca_1.transform(final_data[Grade_normalized.columns])
    E_1 = A_1 / np.sqrt((A_1 * A_1).sum(axis=0))
    B_1 = np.transpose(pca_1.components_)
    
    pca_2 = PCA(n_components=8)
    pca_2.fit(final_data[Exam_1.columns])
    A_2 = pca_2.transform(final_data[Exam_1.columns])
    E_2 = A_2 / np.sqrt((A_2 * A_2).sum(axis=0))
    B_2 = np.transpose(pca_2.components_)
    
    l, m = [], []
    for i in range(1, 9):
        perms = list(itertools.permutations([j for j in range(i)]))
        k, a_ = 0, None
        for a in perms:
            A_3 = np.array([E_2[:, idx] for idx in a]).T
            KK = (E_1[:, :i] * A_3).sum()
            if k < KK / i:
                k = KK / i
                a_ = a
        l.append(a_)
        m.append(KK / i)
    
    l_1, l_2 = [], []
    for i in range(1, 9):
        X_1 = final_data[Grade_normalized.columns] - np.dot(A_1[:, :i], B_1[:i, :])
        C_1 = (X_1 * X_1).sum().sum() / final_data[Grade_normalized.columns].size
        l_1.append(C_1)
        a = l[i-1]
        X_2 = final_data[Exam_1.columns] - np.dot(A_2[:, a], B_2[a, :])
        C_2 = (X_2 * X_2).sum().sum() / final_data[Exam_1.columns].size
        l_2.append(C_2)
    
    plt.figure(figsize=(10, 8), dpi=80)
    plt.plot(range(1, 9), l_1, label="Grade Error")
    plt.plot(range(1, 9), l_2, label="Game Error")
    plt.plot(range(1, 9), np.array(m) - np.array(l_1) - np.array(l_2), label="Best Fit")
    plt.plot(range(1, 9), m, label="Positive Value")
    plt.legend()
    plt.savefig('factor_analysis.png')
    plt.close()
    
    a = (np.array(m) - np.array(l_1) + np.array(l_2)).argmax() + 1
    X = np.array(final_data[Grade_normalized.columns] - np.dot(final_data[Exam_1.columns],
                 np.dot(np.linalg.pinv(B_2[l[a-1], :]), B_1[:a, :])), dtype='float')
    
    plt.figure(figsize=(20, 6), dpi=80)
    sns.heatmap(X, annot=True, linewidths=0.5, cmap="YlGnBu",
                xticklabels=[reverse(i) for i in Grade_normalized.columns])
    plt.savefig('factor_residuals.png')
    plt.close()
    
    return X

# Clustering subjects
def cluster_subjects(Grade_normalized):
    num = 0.7
    B = np.array(Grade_normalized.corr())
    B[B < num] = 0
    
    def add(A, j):
        C = A.copy()
        C.append([j])
        for a in A:
            if B[a, j].sum() == len(a):
                E = a.copy()
                E.append(j)
                C.append(E)
        return C
    
    A = []
    for j in range(B.shape[0]):
        A = add(A, j)
    C = [a for a in A if len(a) > 1]
    
    plt.figure(figsize=(13, 10), dpi=80)
    sns.heatmap(B, annot=True, linewidths=0.5, cmap="YlGnBu",
                yticklabels=[reverse(i) for i in Grade_normalized.columns],
                xticklabels=[reverse(i) for i in Grade_normalized.columns])
    plt.savefig('subject_clustering.png')
    plt.close()
    
    return C

# Main analysis
def main():
    # Preprocess data
    Grade_normalized, ranking_grade, train_Grade, test_num = preprocess_grades()
    Exam = preprocess_exam()
    Exam_1, Exam_2, new_Exam, ranking = compute_metrics(Exam, test_num)
    Exam_Data = cognitive_factors(ranking)
    
    # Combine data
    new_data_frame = pd.concat([Grade_normalized, Exam_Data], axis=1)
    train_data = new_data_frame[~new_data_frame.index.isin(test_num)]
    test_data = new_data_frame[new_data_frame.index.isin(test_num)]
    
    # Correlation analysis
    A = Exam_Data.columns
    B = Grade_normalized.columns
    corr_0 = np.zeros((len(A), len(B)))
    for t, i in enumerate(A):
        for s, j in enumerate(B):
            corr_0[t, s] = Corr(new_data_frame[i], new_data_frame[j])
    
    plt.figure(figsize=(13, 8), dpi=80)
    sns.heatmap(corr_0, annot=True, linewidths=0.5, cmap="YlGnBu",
                xticklabels=[reverse(i) for i in Grade_normalized.columns],
                yticklabels=A)
    plt.savefig('cognitive_grade_correlations.png')
    plt.close()
    
    # Train-test correlations
    corr_2 = np.zeros((len(A), len(B)))
    for t, i in enumerate(A):
        for s, j in enumerate(B):
            corr_2[t, s] = Corr(train_data[i], train_data[j])
    
    corr_3 = np.zeros((len(A), len(B)))
    for t, i in enumerate(A):
        for s, j in enumerate(B):
            corr_3[t, s] = Corr(test_data[i], test_data[j])
    
    # Linear regression
    Y = np.array(train_data[Exam_Data.columns], dtype=float)
    X = np.array(train_data[Grade_normalized.columns], dtype=float)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    Y = np.where(np.isnan(Y), np.nanmean(Y, axis=0), Y)
    A = np.linalg.lstsq(X, Y, rcond=None)[0]
    v = Y - np.dot(X, A)
    train_r2 = 1 - (v * v).sum() / (Y * Y).sum() if (Y * Y).sum() > 0 else 0
    
    y = np.array(test_data[Exam_Data.columns], dtype=float)
    x = np.array(test_data[Grade_normalized.columns], dtype=float)
    x = np.where(np.isnan(x), np.nanmean(x, axis=0), x)
    y = np.where(np.isnan(y), np.nanmean(y, axis=0), y)
    v = y - np.dot(x, A)
    test_r2 = 1 - (v * v).sum() / (y * y).sum() if (y * y).sum() > 0 else 0
    
    # Factor analysis
    X_residuals = factor_analysis(Grade_normalized, Exam_1)
    
    # Subject clustering
    clusters = cluster_subjects(Grade_normalized)
    
    # Save results
    results = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cognitive_grade_correlations': corr_0,
        'subject_clusters': clusters
    }
    return results

if __name__ == "__main__":
    results = main()
    print(f"Train R²: {results['train_r2']:.4f}")
    print(f"Test R²: {results['test_r2']:.4f}")
    print("Subject Clusters:", results['subject_clusters'])
