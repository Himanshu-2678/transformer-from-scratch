# stage10_analysis/tasks/test_tasks.py


from stage10_analysis.tasks.reverse import generate_reverse_task

src, tgt_in, tgt_out = generate_reverse_task(2, 5, 20)

print("SRC:\n", src)
print("TGT_IN:\n", tgt_in)
print("TGT_OUT:\n", tgt_out)
