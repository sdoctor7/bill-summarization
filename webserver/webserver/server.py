import os
from sqlalchemy import *
from sqlalchemy.pool import NullPool
from flask import Flask, request, render_template, g, redirect, Response
from models import *

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)


@app.route('/')
def ex_result():
    if os.path.isfile('ex_result.out'):
        with open("ex_result.out", "r", encoding="utf8") as f:
            ex_content = f.read()
    else:
        ex_content = ""

    if os.path.isfile('ab_result.out'):
        with open("ab_result.out", "r", encoding="utf8") as f:
            ab_content = f.read()
    else:
        ab_content = ""
    return render_template('Index.html',**locals())


@app.route('/inputtext',methods=['POST'])
def input_text():
    addtext = request.form['addtext']
    with open("raw_test.txt","w") as f:
        f.write(addtext)
    cleaning_bill, bill_lengths = split_file(addtext)
    print(request.form['action'])
    if request.form['action'] == 'reset':
        with open("out_b.out", "w", encoding="utf8") as f:
            f.write("")
        with open("raw_test.txt","w") as f:
            f.write("")
       # os.remove("input.txt")
        return redirect('/')
    if request.form['action'] == 'enter':
        return redirect('/')

	
	
@app.route('/extractive',methods=['POST'])
def extractive_models():	
    models=request.form['models']
    # test
    with open("out_b.out", 'r', encoding="utf8") as f:
        cleaning_bill = f.read()
    with open("out_b.out", 'r', encoding="utf8") as f:
        bill_lengths = len(f.readlines())
        print(bill_lengths)
    if models == 'KL-Sum':
        summary_KL = run_sumy(text = cleaning_bill, algo='KL')
        summary_KL_clean = ' '.join([str(sentence) for sentence in summary_KL])
        with open("ex_result.out", "w", encoding="utf8") as f:
            f.write(summary_KL_clean)
    else:
        if bill_lengths < 300:
            print("xxxxxxxxxxxxxxxxx")
            summary_LR = run_sumy(text = cleaning_bill, algo='LexRank')
            summary_LR_clean = ' '.join([str(sentence) for sentence in summary_LR])
        else:
            summary_LR_clean = 'Exceed 300 sentences'
        with open("ex_result.out", "w", encoding="utf8") as f:
            f.write(summary_LR_clean)
    print(request.form['action'])
    if request.form['action'] == 'Remove Output':
        os.remove("ex_result.out")
        return redirect('/')
    if request.form['action'] == 'enter':
        return redirect('/')

@app.route('/abstractive',methods=['POST'])
def abstractive_models():
    # test
    with open("raw_test.txt", 'r', encoding="utf8") as f:
        cleaning_bill = f.read()
    # get number and year of bill
    try:
        relist = re.findall(r'\[(S. .*?) Introduced|\[(H.R. .*?) Introduced', cleaning_bill)[0]
        number = [_ for _ in relist if _][0]  
        year = int(re.findall(r'\[Congressional Bills (\d+)th Congress\]', cleaning_bill)[0])
        number = number.replace(".","").replace(" ","") 
        print("number is %s, year is %i"%(number,year))
        test = pd.read_csv("../test_113_114_115.csv")
        index = test.loc[(test.Number == number) & (test.Congress == year)].index[0]
        print("index : %i"%index)
        with open(os.path.join("../decoded/" + str(index).zfill(6) + "_decoded.txt"), "r", encoding="utf8") as f:
            temp = f.read()
    except:
        temp = "File not in test set. Inaccurate!"
    with open("ab_result.out", "w", encoding="utf8") as f:
            f.write(temp)
    if request.form['action'] == 'Remove Output':
        os.remove("ab_result.out")
        return redirect('/')
    if request.form['action'] == 'enter':
        return redirect('/')



if __name__ == "__main__":
  import click

  @click.command()
  @click.option('--debug', is_flag=True)
  @click.option('--threaded', is_flag=True)
  @click.argument('HOST', default='0.0.0.0')
  @click.argument('PORT', default=8111, type=int)
  def run(debug, threaded, host, port):
    """
    This function handles command line parameters.
    Run the server using
        python server.py
    Show the help text using
        python server.py --help
    """

    HOST, PORT = host, port
    print ("running on %s:%d" % (HOST, PORT))
    app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)


  run()
