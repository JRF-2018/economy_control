# economy_control

<!-- Time-stamp: "2022-05-17T10:30:17Z" -->

村田安雄『動的経済システムの最適制御』の検算＆シミュレーション。

今のところ第2章、第4章に関するシミュレーションと第3章2節b の検算のみ行っ
ている。Google Colab で Python のノートブックを作成して行った。

「検算」の他に scipy.optimize.minimize を使って力技で最適化してみる実
験もしている。

この「プロジェクト」のために、↓も作った。

《JRF-2018/sympy_matrix_tools: Some tools for sympy matrices.》  
https://github.com/JRF-2018/sympy_matrix_tools

本当はこういった検算・証明は専門の theorem prover を使ってやるべきであ
る。しかし、無料で誰でも使える Google Colab & Python で示せればそれは
それで価値があるのではないかと考えた。

なお、*.ipynb と同じ名前を持った *.py は *.ipynb から jupytext を使っ
て生成した。それらは diff がしやすいように作った管理用の Python ファイ
ルである。普通は無視していただいてかまわない。


## 言い訳

本はかなりマイナーな本で、今は絶版状態らしく、手に入れるのはとても難し
いだろう。大学図書館なら、あるところにはあるかもしれない。

その本を持っていてそれを読みながらが前提のため、かなり対象となるユーザー
が限られると思われる。

他の方は、基本的にはこういうこともできるんだという私の自己アピールのよ
うなものと受け取っていただければよいかと思う。私としては、この本のこと
をとてもスゴイと思っていて、この本のことをもっと理解するために何かした
いと思ったのが、本当の動機である。

↓に最初にサッと目を通した後の感想を書いている。

《村田安雄『動的経済システムの最適制御』に目を通した。リカッチ方程式と
かシュタッケルベルク解とか、離散型最大原理・離散型ハミルトニアンなど様々
な概念が出てきて、それをググったりすることでいろいろ知ることができた。 - JRF のひとこと》  
http://jrf.cocolog-nifty.com/statuses/2022/04/post-675aa7.html

また、↓にここに書ききれない裏話などを載せている。

《村田安雄『動的経済システムの最適制御』の検算＆シミュレーションを少し行った。今のところ第2章、第4章、第3章2節b の検算のみで、Google Colab で Python のノートブックを作成して行った。 - JRF のひとこと》  
http://jrf.cocolog-nifty.com/statuses/2022/05/post-3b108d.html


## TODO

今後、他の章や節についても何かやるかもしれない。

ただ、SymPy の限界もある。sympy_matrix_tools で行列計算の限界は突破し
たわけであるが、さらに確率がらみとなると私の手にはあまる。

例えば第3章7節などをしようと思ったら、行列と確率の両方が絡んでくる。実
は、SymPy は今のところ (1,1) の行列とスカラーを区別してしまう。第3章2
節b はその縛りがあっても何とかなったが、7節はそうはいかない感じだ。

最新の SymPy には、RandomSymbol や RandomMatrixSymbol や
RandomIndexedSymbol があり、かなりのことはできるようになっている。 た
だ、RandomMatrixFunction みたいなものはさすがにない。それでも、
epsilon(t) みたいな MatrixFunction を定義しておいて、あとから、
RandomMatrixSymbol の epsilon_T を代入してその都度ごまかすみたいなこと
はできるのかもしれない。

しかし、以下のようなコードを見ていただきたい。

```python
>>> from sympy import *
>>> from sympy.stats import Expectation, Variance
>>> from sympy.stats.rv import RandomSymbol, RandomMatrixSymbol

>>> epsilon = RandomSymbol("epsilon")
>>> zeta = RandomMatrixSymbol("zeta", 2, 1)
>>> A = MatrixSymbol("A", 2, 2)

>>> Expectation(epsilon)
Expectation(epsilon)
>>> Expectation(2 * epsilon).expand()
2*Expectation(epsilon)
>>> Expectation(A * zeta).expand()
A*ExpectationMatrix(zeta)
>>> Expectation(zeta.T * A).expand()
ExpectationMatrix(zeta.T*A)
>>> Expectation(epsilon * Identity(1)).expand()
ExpectationMatrix(epsilon*I)

```

この最後の二つの結果は zeta.T * Expectation(A) と Expectation(epsilon)
* I になって欲しいがそうならないのである。

Expectation にがんばってもらいたいところだが、SymPy の開発者でないので
わからない部分も多く、そういうことができるのはまだまだ先のようである。

また、(1,1) 行列がスカラーとして扱えるようになってもらいたいところだが、
Issues の議論を読むと、(1,1) 行列をスカラーと区別することは「仕様」ら
しく、この点は numpy でも同じなので、そこは期待しても無駄なのかもしれ
ない…。

確率が出て来ない第9章は導出をするのは面倒くさいが、式から数値を得るこ
とぐらいは今の SymPy でも検算できるかもしれないとチラと思ったのだが、
ここも (1,1) 行列がスカラーとして扱えないと(不可能ではないかもしれない
が)面倒くさそうだ。numpy だけで検算するなら容易かもしれないが…。

ただ、ここで述べたようなことがすべての章に必要かというとそうではないだ
ろう。そういう部分は今後の課題としたい。

あと、sympy_matrix_tools で、Sum など数列を扱う部分のテストが全々でき
ていないのは少し心残りである。この本の検算のどこかで使うかと思っていた
が、(今のところ)使わなかった。それを試すのも今後の課題としたい。行列に
関する部分は、今回の検算に使うことでかなりバグが見つかったから、数列に
関する部分もきっとバグがあるだろう。


## Author

JRF ( http://jrf.cocolog-nifty.com/statuses )

The author is Japanese.  I would be grateful if you could ask in Japanese.


## License

MIT License.
