@echo Visualizing

for %%a in (03-12_10-12-27,03-12_10-17-53,03-12_10-22-59) do (
    for %%b in (node_domain,time_domain) do (
        echo %%b
        CALL python run.py visual --file_name=%%a --visual_type=%%b
    ) > .\log\%%a_%%b.txt
)

PAUSE