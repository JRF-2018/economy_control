# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="-s-r8b183fe8"
# # 村田安雄『動的経済システムの最適制御』第4章のシミュレーション
#
# (Version: 0.0.1)
#
# 「第4章 ライフサイクル理論による消費経路」について本にシミュレーション結果があるが、それを「追試」してみる。
#
# 追試の他に scipy.optimize.minimize を使って力技で最適化してみた結果も見てみる。

# + id="SaByRl0m7_GS"
# %matplotlib inline

# + colab={"base_uri": "https://localhost:8080/"} id="jiU2987p_niD" outputId="ad7dcf47-5375-4c51-9055-21fc2be6a4b3"
from time import gmtime, strftime
print("Time-stamp: <%s>" % strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()))

# + id="4LjLWeA8X21F"
import matplotlib.pyplot as plt
import scipy.optimize
import numpy as np

# + [markdown] id="n2aKqnEyYCmz"
# パラメータを本から書き写す。$\sigma^2$ は p.74 から。

# + id="htfEoe5SX_ss"
R = 65
sigma_sq = 0.1625567

# + [markdown] id="kSVjNZGsYWZE"
# p.76 より、基準型(standard)、代替型(alternative)、ケースI、ケースII。

# + id="UZ7UfaRYYZzK"
case_I = {
    'T': 85,
    'p': 0.00652
}

case_II = {
    'T': 90,
    'p': 0.00838
}

standard = {
    'gamma': 3,
    'r': 0.04,
    'theta': 0.01,
    'tau': 0.3,
    'k': 0.5,
    'A22': 200,
    'z': 100,
    'F65': 0
}

alternative = {
    'gamma': 4,
    'r': 0.05,
    'theta': 0.02,
    'tau': 0.2,
    'k': 50,
    'A22': 400,
    'z': 70,
    'F65': 1000
}

# + [markdown] id="R-vCzuOmYd8j"
# p.75 より、年齢別平均所得。

# + id="DVOyp4o8YcmU"
y = [0] * 22 + [
    154.7, # 22
    162.0, # 23
    168.4, # 24
    175.0, # 25
    182.4, # 26
    191.9, # 27
    201.0, # 28
    209.0, # 29
    217.3, # 30
    226.2, # 31
    235.4, # 32

    243.1, # 33
    251.0, # 34
    258.2, # 35
    267.1, # 36
    275.7, # 37
    284.0, # 38
    291.2, # 39
    298.9, # 40
    304.4, # 41
    310.5, # 42
    316.0, # 43

    319.7, # 44
    322.4, # 45
    325.4, # 46
    328.1, # 47
    326.6, # 48
    325.1, # 49
    323.8, # 50
    322.2, # 51
    321.7, # 52
    315.8, # 53
    308.4, # 54
    
    301.1, # 55
    292.1, # 56
    281.0, # 57
    267.0, # 58
    254.0, # 59
    243.5, # 60
    234.8, # 61
    228.5, # 62
    221.0, # 63
    214.9, # 64
    210.0, # 65
]

y_R_orig = y[R]


# + [markdown] id="g1vEERwoYslM"
# パラメータはグローバル変数に代入して使う。

# + id="wQXOaID-Yl31"
def reset_to_standard ():
    GL = globals()
    for n, v in standard.items():
        GL[n] = v
    for n, v in case_I.items():
        GL[n] = v

def set_case_II ():
    GL = globals()
    for n, v in case_II.items():
        GL[n] = v

def set_alternative (*args):
    GL = globals()
    for n in args:
        GL[n] = alternative[n]


# + [markdown] id="KrvPfjvEY0hE"
# パラメータの変更後、常に次の処理をしなければならない。

# + id="1I39Oi2aYyyE"
def set_params ():
    GL = globals()
    GL['rho'] = 1 + r
    GL['alpha'] = (1 + theta) ** -1
    y[R] = y_R_orig + F65


# + [markdown] id="hLxUXw5nY7FF"
# まずは基準型にセットしておく。

# + id="lY3UfT6TY50k"
reset_to_standard()
set_params()


# + [markdown] id="Zfvwu4mtZCyb"
# $C_t$ と $A_t$ を求める方法だが、まず $A_{T+1}$ がわかっているところから $C_T$ を求め、そこから $A_T$ が求まる。次いで $t \ge R + 1$ の間は $A_{t+1}$ と $C_{t+1}$ から $C_t$ と $A_t$ が求まる。そして $C_R$ と $A_R$ が特別に求まった後、今度は逆に $A_{22}$ からはじめてそこから $C_{22}$ が求まり、$t \le R -1$ の間は、$C_t$ と $A_t$ から $C_{t+1}$ と $A_{t+1}$ が求まることになる。問題はパラメータとして $A_{22}$ は与えられるが、$A_{T+1}$ は与えられないという点である。おそらくこの本の著者の手元にはそれを解析的に求める方法があったものと思われる。しかし、私にはその方法がわからない。
#
# そこで scipy.optimize.root を使って適当な $A_{T+1}$ を求めるという方針にする。上の方法でやると、$A_{T+1}$ と $A_{22}$ からは $A_R$ において二通りの値が得られる。その差が 0 となるような $A_{T+1}$ を求めるのだ。
#
# そこで $A_{T+1}$ が与えられたときの $C_t$ と $A_t$ の列および $A_R$ の二つの値の差を出力する関数をまず定義する。それが以下になる。

# + id="jZgJDG2RY_rD"
def get_Cts_Ats (A_Tp1):
    try:
        A_Tp1 = A_Tp1[0]
    except:
        pass
    Ats = [A_Tp1]
    Cts = []
    A_tp1 = A_Tp1
    # 式(25)
    C_T = ((rho * alpha * k * (((1 + p) ** (T - R)) -1)) ** (-1/gamma)) * A_Tp1
    # 式(2')
    A_T = (rho ** -1) * A_Tp1 + C_T - z
    Ats.append(A_T)
    Cts.append(C_T)
    C_tp1 = C_T
    A_tp1 = A_T
    t = T - 1
    while t >= R + 1:
        # 式(25) と 式(26) の間の式
        C_t = ((
            ((1 + p) ** (R - t - 1)) * (C_tp1 ** - gamma)
            + (((1 + p) ** (t - R)) - 1) * k * (A_tp1 ** - gamma)
        ) * rho * alpha) ** (- 1/gamma)
        # 式(2')
        A_t = (rho ** -1) * A_tp1 + C_t - z
        Ats.append(A_t)
        Cts.append(C_t)
        C_tp1 = C_t
        A_tp1 = A_t
        t = t - 1
    assert t == R
    A_Rp1 = A_tp1
    # 式(27') の下の式
    C_R = (rho * alpha * ((1 + p) ** -1) * (C_tp1 ** - gamma)) ** (-1/gamma)
    # 式(1')
    A_R = (rho ** -1) * A_tp1 + C_R - (1 - tau) * y[R]
    Ats.append(A_R)
    Cts.append(C_R)
    rAts = list(reversed(Ats))
    rCts = list(reversed(Cts))
    Ats = []
    Cts = []
    t = 22
    A_t = A22
    while t <= R - 1:
        # 式(36)
        L_t = A_t \
            + (1 - tau) * sum([y[t + i] * (rho ** - i)
                               for i in range(R - t + 1)]) \
            - A_Rp1 * (rho ** (t - R - 1))
        # 式(34)
        g = ((rho ** (1 - gamma)) * alpha) ** (1/gamma)
        # 式(33)
        G_tp1 = (1 - g ** (R - (t + 1) + 1)) / (1 - g)
        # 式(40)
        C_t = L_t / (1 + ((rho ** (1 - gamma)) * alpha * (G_tp1 ** gamma)
                          * (1 + (1/2) * gamma * (gamma + 1) * sigma_sq))
                     ** (1/gamma))
        Ats.append(A_t)
        Cts.append(C_t)
        # 式(1)
        A_t = (A_t + (1 - tau) * y[t] - C_t) * rho
        t = t + 1
    d65 = A_R - A_t
    return Cts + rCts, Ats + rAts, d65



# + [markdown] id="PJkEI6HXcFAk"
# ところでこの関数がまともに動くまでデバッグするには苦労した。もちろん私が産んだケアレスミスもあったが、本にも誤植があった。
#
# 一つは式(40)において、大カッコ「[」が 1 + の後、$\rho^{1-\gamma}$ の前に抜けていたこと。まぁ、これは大したことない。
#
# もう一つは式(34) いおいて、$\gamma$ であるべきところが $r$ に置き換わっていたこと。ありがちな誤植だが、これで結果がかなり変わっていた。

# + [markdown] id="QJmmweiEdJbE"
# さてここで root finding を実際してみよう。

# + colab={"base_uri": "https://localhost:8080/"} id="AUzXLGSMdQB9" outputId="56d5659e-ff98-4865-9615-a379fc806c23"
def score_A_Tp1 (A_Tp1):
    Cts, Ats, d65 = get_Cts_Ats(A_Tp1)
    return d65

sol = scipy.optimize.root(score_A_Tp1, A22)
print(sol)

# + [markdown] id="AnqaVKVPdf7F"
# ちゃんと求まったようだ。p.78 の数値の $A_{T+1}$ は 141 なのであってる感じである。一応、$C_t$ と $A_t$ の列も見ておこう。

# + colab={"base_uri": "https://localhost:8080/"} id="9oolNRS7d13t" outputId="7502118f-3be7-40ee-c5ef-b8fa19f71f28"
Cts, Ats, d65 = get_Cts_Ats(sol.x)
Cts

# + colab={"base_uri": "https://localhost:8080/"} id="QTH2c0rbdXA-" outputId="ca551bf6-e7f4-4a6a-c803-5c5cfecc74c2"
Ats

# + colab={"base_uri": "https://localhost:8080/"} id="Ofmb9uARd_zP" outputId="645705f9-8e17-4243-e7fe-b23ac04678a2"
d65

# + [markdown] id="GH0bwMyQggSV"
# ここで得られた Cts Ats に名前を付けておく。

# + id="gsYUuTdjgebu"
Ats0 = Ats
Cts0 = Cts


# + [markdown] id="MTqTX7kWeSKe"
# さてここからは複雑な式を解析的に求めるのではなく、力技で最適化したら、同じ結果が出るのかというのを調べてみたい。そのために必要な定義を書き写していく。$C_t$ の値を求めれば $A_t$ の値は簡単に出るので、$C_t$ の値をいろいろな方法で出してそれが上の Cts0 と等しいか見ていく。
#
# Cts のベクトルの正しさは差の二乗和によってみるのでそれ用の関数 ssd を定義しておく。

# + id="oNpsANu-eB8-"
def ssd (a, b): # Sum of Squared Difference
    a = np.array(a)
    b = np.array(b)
    return np.sum((a - b) ** 2)


# + [markdown] id="P3JxTZiLfT7m"
# 基本的な定義・式を書き写す。

# + id="BLMXYS5ZfIjm"
# 式(21)
def U (C):
    return ((1 - gamma) ** -1) * (C ** (1 - gamma))

# 式(22)
def W (A):
    return k * ((1 - gamma) ** -1) * (A ** (1 - gamma))

# 式(23)
def P (t):
    if t < R:
        return 0
    #assert t <= T
    return 1 - ((1 + p) ** (R - t))

# 式(3)の前の式。
def Phi(t1, t2):
    assert t2 >= t1
    return np.prod([(1 - P(t)) for t in range(t1, t2 + 1)])

# 式(3)
def score_Et_UT (t, Cts, Ats):
    s = 0
    for i in range(T - t + 1):
        Pti = P(t + i)
        s += Phi(t, t + i) * (U(Cts[t + i - 22])
                                + ((Pti / (1 - Pti)) * W(Ats[t + 1 + i - 22])
                                   * alpha)) * (alpha ** i)
    return s


# + [markdown] id="GqfwCBo6fll9"
# Cts が決まっているなら Ats は簡単に求まる。

# + id="lkQRW4-ofdLu"
def get_Ats (Cts):
    Ats = [A22]
    A_t = A22
    t = 22
    while t <= R:
        # 式(1)
        A_tp1 = (A_t + (1 - tau) * y[t] - Cts[t - 22]) * rho
        A_t = A_tp1
        Ats.append(A_t)
        t = t + 1
    while t <= T:
        # 式(2)
        A_tp1 = (A_t + z - Cts[t - 22]) * rho
        A_t = A_tp1
        Ats.append(A_t)
        t = t + 1
    return Ats


# + [markdown] id="XsNj7goLgUQP"
# いちおう検算しておく。

# + colab={"base_uri": "https://localhost:8080/"} id="podjlym8gQTG" outputId="6118e815-913f-4367-8fce-cb99652b7e80"
Ats = get_Ats(Cts0)
ssd(Ats, Ats0)


# + [markdown] id="QruYAr6LfsNG"
# 最適化は Cts の長い列をまるごと求める形になる。
#
# 最適化の途中ときどき、Ats や Cts がマイナスになることがあった。そこでそれらの値が起こらないよう制約を付ける。bounds や constraints を指定して最適化手法は SLSQP を使う。
#
# Ats の制約は次のようにした。

# + id="NdDtn6xxfkvd"
def Ats_cons (Cts):
    Ats = get_Ats(Cts)
    return sum([i for i in Ats if i < 0])


# + [markdown] id="7rsp_MQOg1_l"
# さて、最適化はどういうことに関して行っているのか。実はこの本には「目的関数」がこの章では直接書かれていない。そこでまずは最も簡単な候補として、各 $t$ における効用を純粋に足し合わせたものを考える。それを Cts1 とする。なお、scipy.optimize.minimize は最小化でこれは効用の最大化であるため、スコアをマイナスにする必要があることに注意すること。

# + colab={"base_uri": "https://localhost:8080/"} id="y2IToKzLgsEG" outputId="869d93bb-f01e-4cc1-d642-2b8f272338a3"
def score_simple_sum (Cts):
    Ats = get_Ats(Cts)
    s = 0
    for t in range(22, T+1):
        Pt = P(t)
        # 式(2) の後の式
        s += (1 - Pt) * U(Cts[t - 22]) + Pt * W(Ats[t + 1 - 22]) * alpha
    return -s

res = scipy.optimize.minimize(score_simple_sum, [10] * (T - 22 + 1),
                              bounds=([(0, np.inf)] * (T - 22 + 1)),
                              constraints=(
                                  {'type': 'ineq', 'fun': Ats_cons},
                              ), method="SLSQP")
res

# + colab={"base_uri": "https://localhost:8080/"} id="tFFtgl7LhUK_" outputId="509770e0-34a6-4471-bee7-c64a24340776"
Cts1 = res.x
Ats1 = get_Ats(Cts1)
Ats1


# + colab={"base_uri": "https://localhost:8080/"} id="LdnbsDPWhj_3" outputId="29522ceb-99f1-4a88-efab-6aca35f3c870"
ssd(Cts1, Cts0)


# + [markdown] id="17eMRopIhoeG"
# …ということで、Cts0 とはかなり異なる結果が得られた。ただ消費が毎期ほぼ同じなので、これはこれでまともな結果ではないかという気もする。ただ、資産が残り過ぎか。

# + [markdown] id="m8-U3LdHh6jm"
# 次に 22歳のときの効用を最大化することを考える。$E_{22}U_{T}$ を最大化してみる。

# + colab={"base_uri": "https://localhost:8080/"} id="oAaj1nmWhnXW" outputId="305a60d8-891e-410b-844e-d8357519a285"
def score_E22_UT (Cts):
    return -score_Et_UT(22, Cts, get_Ats(Cts))

res = scipy.optimize.minimize(score_E22_UT, [10] * (T - 22 + 1),
                              bounds=([(0, np.inf)] * (T - 22 + 1)),
                              constraints=(
                                  {'type': 'ineq', 'fun': Ats_cons},
                              ), method="SLSQP")
res

# + colab={"base_uri": "https://localhost:8080/"} id="IALm7-0miki2" outputId="0e3f06b4-5ffe-411e-de94-dcf0c7363a02"
Cts2 = res.x
Ats2 = get_Ats(Cts2)
Ats2

# + colab={"base_uri": "https://localhost:8080/"} id="N6besA4fiqtP" outputId="37b93403-77c8-4c26-ee5d-088c0bfcfa2d"
ssd(Cts2, Cts0)


# + [markdown] id="Re4Jus23iv3-"
# …ということで、これは22歳のときのみの判断なため、高齢になったときの消費を過少評価しているようだ。当然 Cts0 とも大きく違う。

# + [markdown] id="Lx4D7Jm4jDH-"
# ここでどうすればよいか詰まった。22歳のときだけでないようにすればいいのだから、$E_tU_T$ を $t$ に関して足し合わせればいいのかとも考えた。しかし、それは求めるのが重すぎるだけで、どうも正しい結果とは思えない。一応コメントアウトしてそのコードだけは載せておこう。

# + id="pbM0Wb0hjdFn"
# def score_sum_Et_UT (Cts):
#     Ats = get_Ats(Cts)
#     s = 0
#     for t in range(22, T + 1):
#         s += score_Et_UT(t, Cts, Ats)
#     return -s
#
# res = scipy.optimize.minimize(score_sum_Et_UT, [10] * (T - 22 + 1),
#                               bounds=([(0, np.inf)] * (T - 22 + 1)),
#                               constraints=(
#                                   {'type': 'ineq', 'fun': Ats_cons},
#                               ), method="SLSQP")
# res

# + id="WdMbmHk3iuJJ"
# Cts3 = res.x
# Ats3 = get_Ats(Cts3)
# Ats3

# + id="XXhPcTFLjk1v"
# ssd(Cts3, Cts0)

# + [markdown] id="qhean9Gnjp-I"
# 本ではベルマン方程式を使っていく。ではそのベルマン方程式にできるだけ忠実にやってみてはどうかと考えた。それが以下になる。

# + colab={"base_uri": "https://localhost:8080/"} id="XnPTLit6joVW" outputId="63235b52-5fc1-4a2d-ec85-509bc7441958"
def score_V (Cts):
    Ats = get_Ats(Cts)
    V = 0
    t = T
    while t >= 22:
        V = (1 - P(t)) * U(Cts[t - 22]) + P(t) * W(Ats[t + 1 - 22]) * alpha \
            + (1 - P(t)) * alpha * V
        t = t - 1
    return -V

res = scipy.optimize.minimize(score_V, [10] * (T - 22 + 1),
                              bounds=([(0, np.inf)] * (T - 22 + 1)),
                              constraints=(
                                  {'type': 'ineq', 'fun': Ats_cons},
                              ), method="SLSQP")
res

# + colab={"base_uri": "https://localhost:8080/"} id="LMbYRKtUj6VB" outputId="7c2ebdee-aeab-4ef7-b03a-f1c44539c31c"
Cts4 = res.x
Ats4 = get_Ats(Cts4)
Ats4

# + colab={"base_uri": "https://localhost:8080/"} id="-TWoG5C2kAqP" outputId="5c28a1d1-8d98-4177-86ec-1386abe9bd21"
ssd(Cts4, Cts0)

# + [markdown] id="UNiAcnFKkFDg"
# ところがどうもこの CTs4 は Cts2 と同じものらしい。

# + colab={"base_uri": "https://localhost:8080/"} id="5dK4s2sLkDtf" outputId="0013facf-c0d2-424f-c6a1-562ff6df60d5"
ssd(Cts4, Cts2)


# + [markdown] id="eROn6Pl0kRaP"
# …ということで、ベルマン方程式って何を求めてるのだ？ と考えることになった。$t$ 期ごとに最適化するということでそれは Cts を一気に最適化するのとはまた違うのではないか？ …そう考えるに至った。
#
# 逐次に最適化していく…というアイデアを推し進めてみよう。22歳のときには高齢の影響をほぼ無視するが、高齢になれば今の自分から見てさらに高齢になることを無視しなくなるだろう。
#
# そこでまず $E_{22}U_T$ で 22歳のときは $C_{22}$ を判定するが、23歳になればそれはそれで $E_{23}U_T$ から $C_{23}$ を判定すると考えればどうか。以降 $C_{22}$ から $C_{t-1}$ まで決まっている前提で、逐次 $E_tU_T$ から $C_t$ を求めればいいのではないか。
#
# そういう方針でやったのが次のものである。これは時間がかかる…。

# + colab={"base_uri": "https://localhost:8080/"} id="cGSDvotKl0pY" outputId="9cf0cba3-5f1b-45ea-d263-b1f646dcd8c5"
def score_Et_UT_2 (Cts2, Cts1):
    Cts = list(Cts1) + list(Cts2)
    Ats = get_Ats(Cts)
    return - score_Et_UT(22 + len(Cts1), Cts, Ats)

def Ats_cons_2 (Cts2, Cts1):
    Ats = get_Ats(list(Cts1) + list(Cts2))
    return sum([i for i in Ats if i < 0])

Cts = []
for t in range(22, T + 1):
    res = scipy.optimize.minimize(score_Et_UT_2, [10] * (T - 22 + 1 - len(Cts)),
                                  args=(Cts,),
                                  bounds=([(0, np.inf)] * (T - 22 + 1 - len(Cts))),
                                  constraints=(
                                      {'type': 'ineq', 'fun': Ats_cons_2,
                                       'args': (Cts,)},
                                  ), method="SLSQP")
    Cts.append(res.x[0])
    print(t, res.success)

Cts5 = Cts
Cts5

# + colab={"base_uri": "https://localhost:8080/"} id="f7xLnH3Il8Gg" outputId="74d439f2-8908-4f80-9cef-2736fccd73a3"
Ats5 = get_Ats(Cts5)
Ats5

# + colab={"base_uri": "https://localhost:8080/"} id="mwBYHIehmeZw" outputId="545a0c18-d4c3-4cb9-c948-a93d9647c78d"
ssd(Cts5, Cts0)

# + [markdown] id="Bd8ZC3Anmj7Q"
# …ということで、これはかなりイイトコロにいってるのだと私は思うのだが、Cts0 とはだいぶ違った結果になった。ただ、Cts0 は近似を使っているのでもしかすると Cts5 のほうが Cts0 より良い結果なのかもしれない。
#
# 気になるところとしては $C_T$ が 10 程度とかなり低いのが何かおかしいのかと思うが、今のところバグらしきものはなさそうで謎である。
#
# 結果的に ssd は Cts1 が一番小さく、なんだかなぁ…という感じである。

# + [markdown] id="-joQ8ZTanJka"
# さて一応これまでのところを図示しておこう。

# + colab={"base_uri": "https://localhost:8080/", "height": 281} id="IWAE6TOLnPT4" outputId="9242a54c-e611-477d-dd0a-e83b06f44861"
fig, ax = plt.subplots()
t1 = list(range(22, T + 1))
ax.plot(t1, Cts0, label='0')
ax.plot(t1, Cts1, label='1')
ax.plot(t1, Cts2, label='2')
ax.plot(t1, Cts4, label='4')
ax.plot(t1, Cts5, label='5')
ax.set_xlabel('$t$')
ax.set_ylabel('$C_t$')
ax.legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 281} id="NLiwVfUymi25" outputId="1d495482-5789-4e70-d1d7-c6aa6c567248"
fig, ax = plt.subplots()
t1 = list(range(22, T + 2))
ax.plot(t1, Ats0, label='0')
ax.plot(t1, Ats1, label='1')
ax.plot(t1, Ats2, label='2')
ax.plot(t1, Ats4, label='4')
ax.plot(t1, Ats5, label='5')
ax.set_xlabel('$t$')
ax.set_ylabel('$A_t$')
ax.legend()
plt.show()

# + [markdown] id="E13vbK17swAA"
# さてここからは単純に図4-1a から図4-10b まで「検算」して図示していこう。

# + id="0XWDc1KMnZmY"
Yts = y[22:R+1] + [z] * (T - R)

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="Nkg4M6W3tJE7" outputId="27cf30ae-1538-4933-a02c-3499f0219bd5"
fig, ax = plt.subplots()
t1 = list(range(22, T + 1))
ax.plot(t1, Yts, label='$Y$')
ax.plot(t1, Cts0, label='$C$')
ax.set_xlabel('$t$')
ax.set_ylabel('$C$, $Y$')
ax.set_title('Fig. 4-1a')
ax.legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="TpBReVuKtMxD" outputId="a1365953-a148-45fa-ece1-7d36b637c042"
fig, ax = plt.subplots()
t1 = list(range(22, T + 1))
t2 = list(range(22, T + 2))
ax.plot(t2, Ats0, label='$A$')
ax.plot(t1, Cts0, label='$C$')
ax.set_xlabel('$t$')
ax.set_ylabel('$A$, $C$')
ax.set_title('Fig. 4-1b')
ax.legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/"} id="Q8ucpdsTuzA7" outputId="1f1b548e-a77e-4d0b-c14e-69cd2c29eb6f"
reset_to_standard()
set_alternative('r')
set_params()

sol = scipy.optimize.root(score_A_Tp1, A22)
sol.success

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="bo6OvIZRvdKb" outputId="9817bb9b-18d1-4da2-cc2d-d37e58afeeb1"
Cts1, Ats1, _ = get_Cts_Ats(sol.x)
fig, ax = plt.subplots()
t1 = list(range(22, T + 1))
ax.plot(t1, Yts, label='$Y$')
ax.plot(t1, Cts1, label='$C$')
ax.set_xlabel('$t$')
ax.set_ylabel('$C$, $Y$')
ax.set_title('Fig. 4-2')
ax.legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/"} id="l41jzaR_wNoz" outputId="0b580f98-07b6-470b-8cf9-39f96573a626"
reset_to_standard()
set_alternative('tau')
set_params()

sol = scipy.optimize.root(score_A_Tp1, A22)
sol.success

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="krYM7Jh-weoz" outputId="ae9d1076-29a5-4cfa-d05c-7d5ce2ccdd87"
Cts1, Ats1, _ = get_Cts_Ats(sol.x)
fig, ax = plt.subplots()
t1 = list(range(22, T + 1))
ax.plot(t1, Yts, label='$Y$')
ax.plot(t1, Cts1, label='$C$')
ax.set_xlabel('$t$')
ax.set_ylabel('$C$, $Y$')
ax.set_title('Fig. 4-3')
ax.legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/"} id="JtNJ4bsSwksb" outputId="f3c11878-8bd1-490e-a642-5020890192bc"
reset_to_standard()
set_alternative('theta')
set_params()

sol = scipy.optimize.root(score_A_Tp1, A22)
sol.success

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="Rr6bI9NCxAtL" outputId="eb1d3974-151e-46ab-fd8d-e350d13df22f"
Cts1, Ats1, _ = get_Cts_Ats(sol.x)
fig, ax = plt.subplots()
t1 = list(range(22, T + 1))
ax.plot(t1, Cts0, label='$θ = 0.01$')
ax.plot(t1, Cts1, label='$θ = 0.02$')
ax.set_xlabel('$t$')
ax.set_ylabel('$C$')
ax.set_title('Fig. 4-4')
ax.legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/"} id="D-Xit8gHxN0L" outputId="0f967271-9565-400a-9477-4935606059c8"
reset_to_standard()
set_alternative('F65')
set_params()

sol = scipy.optimize.root(score_A_Tp1, A22)
sol.success

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="c6t6b1LvxsWz" outputId="fdcaa229-5be0-4e7b-de50-09c92845e8d0"
Cts1, Ats1, _ = get_Cts_Ats(sol.x)
fig, ax = plt.subplots()
t1 = list(range(22, T + 1))
ax.plot(t1, Cts0, label='$F_{65} = 0$')
ax.plot(t1, Cts1, label='$F_{65} = 1000$')
ax.set_xlabel('$t$')
ax.set_ylabel('$C$')
ax.set_title('Fig. 4-5a')
ax.legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="a4ef08j3x6yr" outputId="96fd509d-ad59-4487-f13b-96232efbb2d7"
fig, ax = plt.subplots()
t1 = list(range(22, T + 2))
ax.plot(t1, Ats0, label='$F_{65} = 0$')
ax.plot(t1, Ats1, label='$F_{65} = 1000$')
ax.set_xlabel('$t$')
ax.set_ylabel('$A$')
ax.set_title('Fig. 4-5b')
ax.legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/"} id="TbCyBhCCyFD8" outputId="b56ef5c3-2c66-4aee-a099-986dfb1ac38d"
reset_to_standard()
set_alternative('z')
set_params()

sol = scipy.optimize.root(score_A_Tp1, A22)
sol.success

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="Krd0bkTIyVyc" outputId="da2a2278-af73-486e-ce06-bd16c338a06c"
Cts1, Ats1, _ = get_Cts_Ats(sol.x)
fig, ax = plt.subplots()
t1 = list(range(22, T + 1))
ax.plot(t1, Cts0, label='$z = 100$')
ax.plot(t1, Cts1, label='$z = 70$')
ax.set_xlabel('$t$')
ax.set_ylabel('$C$')
ax.set_title('Fig. 4-6a')
ax.legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="0X0AOQoCygEU" outputId="47e2003c-f8a7-434f-c3c1-0c3926827960"
fig, ax = plt.subplots()
t1 = list(range(22, T + 2))
ax.plot(t1, Ats0, label='$z = 100$')
ax.plot(t1, Ats1, label='$z = 70$')
ax.set_xlabel('$t$')
ax.set_ylabel('$A$')
ax.set_title('Fig. 4-6b')
ax.legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/"} id="ZuBMd9dYypK0" outputId="0e3325fe-c31b-40ea-e031-426a03847316"
reset_to_standard()
set_case_II()
set_params()

sol = scipy.optimize.root(score_A_Tp1, A22)
sol.success

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="Nu4blM1mzBV1" outputId="d1fcef7a-4f07-446c-b192-7a0c43d17887"
Cts1, Ats1, _ = get_Cts_Ats(sol.x)
fig, ax = plt.subplots()
t1 = list(range(22, case_I['T'] + 1))
t2 = list(range(22, case_II['T'] + 1))
ax.plot(t1, Cts0, label='$T = 85$')
ax.plot(t2, Cts1, label='$T = 90$')
ax.set_xlabel('$t$')
ax.set_ylabel('$C$')
ax.set_title('Fig. 4-7a')
ax.legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="CT426FzOzTuU" outputId="8803a6d4-7943-4f66-8148-0f5d43c33788"
fig, ax = plt.subplots()
t1 = list(range(22, case_I['T'] + 2))
t2 = list(range(22, case_II['T'] + 2))
ax.plot(t1, Ats0, label='$T = 85$')
ax.plot(t2, Ats1, label='$T = 90$')
ax.set_xlabel('$t$')
ax.set_ylabel('$A$')
ax.set_title('Fig. 4-7b')
ax.legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/"} id="D5zyWzwwzkSt" outputId="73af47b8-630b-4baf-e812-2bc9ca0b2a0b"
reset_to_standard()
set_case_II()
set_alternative('gamma')
set_params()

sol = scipy.optimize.root(score_A_Tp1, A22)
sol.success

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="PEbtjBcE0aFU" outputId="8214269e-8c74-49c6-edbd-90666190234a"
Cts2, Ats2, _ = get_Cts_Ats(sol.x)
fig, ax = plt.subplots()
t1 = list(range(22, case_II['T'] + 1))
ax.plot(t1, Cts1, label='$γ = 3$')
ax.plot(t1, Cts2, label='$γ = 4$')
ax.set_xlabel('$t$')
ax.set_ylabel('$C$')
ax.set_title('Fig. 4-8a')
ax.legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="TYh_qDSW0b-k" outputId="dd7d7adb-88f3-4d77-a934-ae2a18bf65a4"
fig, ax = plt.subplots()
t1 = list(range(22, case_II['T'] + 2))
ax.plot(t1, Ats1, label='$γ = 3$')
ax.plot(t2, Ats2, label='$γ = 4$')
ax.set_xlabel('$t$')
ax.set_ylabel('$A$')
ax.set_title('Fig. 4-8b')
ax.legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/"} id="DVeLmozI0gcV" outputId="5fdd3d8c-5122-41cd-e1b3-30178fe45967"
reset_to_standard()
set_case_II()
set_alternative('k')
set_params()

sol = scipy.optimize.root(score_A_Tp1, A22)
sol.success

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="LWKK5C0K0uUV" outputId="0f95c133-d4a5-4b39-84eb-5a222097ca13"
Cts2, Ats2, _ = get_Cts_Ats(sol.x)
fig, ax = plt.subplots()
t1 = list(range(22, case_II['T'] + 2))
ax.plot(t1, Ats1, label='$k = 0.5$')
ax.plot(t2, Ats2, label='$k = 50$')
ax.set_xlabel('$t$')
ax.set_ylabel('$A$')
ax.set_title('Fig. 4-9')
ax.legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/"} id="k4F-DPu70wVs" outputId="446eff1c-ba54-4cb3-ff34-09e4e75ea637"
reset_to_standard()
set_case_II()
set_alternative('A22')
set_params()

sol = scipy.optimize.root(score_A_Tp1, A22)
sol.success

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="zh-6yegV1eRc" outputId="9c1a00f3-c7fa-40f0-e717-24f7d0062c40"
Cts2, Ats2, _ = get_Cts_Ats(sol.x)
fig, ax = plt.subplots()
t1 = list(range(22, case_II['T'] + 1))
ax.plot(t1, Cts1, label='$A_{22} = 200$')
ax.plot(t1, Cts2, label='$A_{22} = 400$')
ax.set_xlabel('$t$')
ax.set_ylabel('$C$')
ax.set_title('Fig. 4-10a')
ax.legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="88OzpBTo1gEc" outputId="05a547f9-f8c9-4def-e668-761eb1e7c7a7"
fig, ax = plt.subplots()
t1 = list(range(22, case_II['T'] + 2))
ax.plot(t1, Ats1, label='$A_{22} = 200$')
ax.plot(t2, Ats2, label='$A_{22} = 400$')
ax.set_xlabel('$t$')
ax.set_ylabel('$A$')
ax.set_title('Fig. 4-10b')
ax.legend()
plt.show()

# + [markdown] id="v4GNWw9R17U9"
# ちゃんと「検算」できてるように思う。
