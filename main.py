# sample

# gpu test
if False:
    from src.utils import gpu_test


from examples import basic_usage, demo_log_analysis

# Analyze the created logs
demo_log_analysis.create_sample_logs()

if False:
    # basic usage
    basic_usage.main()
else:
    # log analysis system
    demo_log_analysis.main()
