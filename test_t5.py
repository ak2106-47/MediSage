from t5_rewriter import T5Rewriter

rewriter = T5Rewriter()
while True:
    q = input("User: ")
    print("Rewritten:", rewriter.rewrite(q))
