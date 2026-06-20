import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
q=["Q1\nrarest","Q2","Q3\nsweet spot","Q4\ncommonest"]
s=[0.224,0.480,0.542,0.478]
plt.figure(figsize=(8,5))
bars=plt.bar(q,s,color=["#f85149","#d29922","#2ea043","#d29922"])
for b,v in zip(bars,s): plt.text(b.get_x()+b.get_width()/2,v+0.01,f"{v:.2f}",ha="center",fontsize=12)
plt.axhline(0.9,color="#888",ls="--",label="fully stable")
plt.ylabel("avg stability (higher = replicates)")
plt.xlabel("firing frequency, low → high")
plt.title("Stability vs frequency: rises sharply, peaks mid-high, dips at the top\n(an inverted-U — both extremes are less stable)")
plt.ylim(0,0.65); plt.legend(); plt.tight_layout()
plt.savefig("/Users/oe/rebuild/fig6_invertedU.png",dpi=150)
print("saved")
