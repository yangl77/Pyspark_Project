def cal_run_time(start, end):
    run_time = round(end - start)
    hours = run_time // 3600
    minutes = (run_time - 3600 * hours) // 60
    seconds = run_time - 3600 * hours - 60 * minutes
    print(f'Time used：{hours} hours {minutes} minutes {seconds} seconds')