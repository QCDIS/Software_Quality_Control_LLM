

python run_quality_scan_cli.py --stage Development --path test_notebooks/ --save testing_save.json
Res: works

python run_quality_scan_cli.py --stage Development --path test_notebooks/SFT.ipynb --save testing_save.json
Res: Works but wrong answer

python run_quality_scan_cli.py --stage Maintenance --path test_notebooks/ --save testing_save.json
Res: Works but 2 results fails

python run_quality_scan_cli.py --stage Maintenance --path test_notebooks/SFT.ipynb --save testing_save.json
Res: Not meant to work without a directory


python run_quality_scan_cli.py --stage Maintenance --path test_notebooks/ --save testing_save.json --github https://github.com/yutong0310/jupyter-quality-extension
Res: Worke, maintenance should be accoompanied by github links
Discuss: Directory and github should be the same content? If same, why both of them are needed.

