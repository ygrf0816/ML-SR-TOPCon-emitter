#使用方法
# import python_tcad
# a=python_tcad.pythontcad() 
# a.set_tcad('D:/silvaco/exe/') #设置tcad仿真器的地址
# a.read_in('perovskite.in')  #读取in文件
# a.run() #运行 可选参数output_state，默认为true，会把仿真器中输出的信息显示出来。设置为FALSE就不显示
#命令保存在commands变量中，变量保存在variables中
#虽然写了Athena工艺仿真的代码，但是实际上不能运行，目前只支持atlas。
import subprocess
import os
import re
deckkeywords=['loop','stmt','extract','l.end','tonyplot']
import time
import tcaddata
import tcadmodel
import sys
import threading

def read_line_with_timeout(process, output_list, timeout):
    """
    从子进程读取一行，支持超时。
    """
    def read_output():
        line = process.stdout.readline()
        if line:
            output_list.append(line)

    output = []
    thread = threading.Thread(target=read_output)
    thread.start()
    thread.join(timeout)  # 等待线程完成或超时
    if thread.is_alive():
        print("读取超时！")
        thread.join()  # 清理线程
    return ''.join(output_list)
class pythontcad():
    def __init__(self) -> None:
        self.atlas=''
        self.variables={}
        self.commands=[]
        self.tcad=tcaddata.tcad()

    def set_tcad(self,tcad):
        if tcad[-1]!='\\' or tcad[-1]!='/':
            self.atlas=tcad+'/atlas.exe'
            self.athena=tcad+'/athena.exe'
        else:
            self.atlas=tcad+'atlas.exe'
            self.athena=tcad+'athena.exe'
    def eval_equation(self,str1):
        pattern = r'[-+*/()\d ]+'
        # 使用re.search函数查找匹配的部分
        match = re.search(pattern, str1)
        # 如果找到匹配的部分，表示存在算式
        if match:
            result=re.sub(r'\s*=\s*', '=', str1)
            caozuo=result.split(' ')
            caozuo2=[]
            for i in caozuo:
                if '=' not in i:
                    caozuo2.append(i)
                else:
                    a,b=i.split('=')
                    if re.search(r'[-+*/() ]+', b):
                        try:
                            b=eval(b)
                        except:
                            pass
                    caozuo2.append(f'{a}={b}')
            result=' '.join(caozuo2)
            return result
        else:
            return str1

    def read_in(self,filename):
        variables = {}  # 用于存储变量的字典
        commands = []  # 用于存储命令的列表
        
        with open(filename, 'r',encoding='utf-8') as file:
            lines = file.readlines()
            multiline=False
            multilinecommand=''
            for line in lines:
                line = line.split('#')[0].strip()  # 移除注释和前后的空白字符
                line =re.sub(r'\s+', ' ', line)
                if line.startswith('set'): 
                    parts = line.split()  
                    if len(parts) == 2: 
                        var_name, value = parts[1].split('=')
                        variables[var_name] = value  
                elif multiline==True and not line.endswith('\\'):
                    multiline=False
                    multilinecommand+=line
                    commands.append(multilinecommand)
                    multilinecommand=''
                elif line.endswith('\\'):
                    multiline=True
                    multilinecommand+=line.replace('\\','')
                elif line:  
                    commands.append(line)
        variables=dict(sorted(variables.items(), key=lambda item: len(item[0]), reverse=True))
        vcopy=variables.copy()
        for var_name in variables:
            for var_name2 in vcopy:
                pattern = re.compile(re.escape(f'${var_name2}'), re.IGNORECASE)
                variables[var_name] = pattern.sub(vcopy[var_name2], variables[var_name])
        self.variables=variables
        self.commands=commands

    def run(self,output_state=True):
        starttime=time.time()
        rawcommandlines=self.commands.copy()
        commandlines=[]
        var=self.variables.copy()
        for i1 in rawcommandlines:
            for i2 in var:
                # 使用正则表达式替换，忽略大小写
                pattern1 = re.compile(re.escape(f'${i2}'), re.IGNORECASE)
                i1 = pattern1.sub(var[i2], i1)

                pattern2 = re.compile(re.escape(f"$'{i2}'"), re.IGNORECASE)
                i1 = pattern2.sub(var[i2], i1)

                pattern3 = re.compile(re.escape(f'$"{i2}"'), re.IGNORECASE)
                i1 = pattern3.sub(var[i2], i1)
                # if f'${i2}' in i1:
                #     i1=i1.replace(f'${i2}',var[i2])
                # elif f'$\'{i2}\'' in i1:
                #     i1=i1.replace(f'$\'{i2}\'',var[i2])
                # elif f'$\"{i2}\"' in i1:
                #     i1=i1.replace(f'$\"{i2}\"',var[i2])
            commandlines.append(i1)
        
        
        mode=''
        for i in commandlines:
            keyword=i.split(' ')[0]
            if keyword in deckkeywords:
                continue
            if 'go atlas' in i.lower():
                mode='atlas'
                self.atlas_process=subprocess.Popen(self.atlas,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                # self.run_atlas_start(output_state=output_state)
                continue
            elif 'go athena' in i.lower():
                mode='athena'
                self.athena_process=subprocess.Popen(self.athena,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='ISO-8859-1')
                # self.run_athena_start(output_state=output_state)
                continue
            command=self.eval_equation(i)
            if mode=='atlas':
                self.run_atlas_command(command,output_state=output_state)
            elif mode=='athena':
                self.run_athena_command(command,output_state=output_state)
        totaltime=time.time()-starttime
        print(f'本次仿真用时{totaltime:.2f}s')

    def run_athena_start(self, output_state=True):
        return self._read_until_prompt(self.athena_process, output_state=output_state)

    def _read_until_prompt(self, process, output_state=False):
        """读取子进程输出，直到出现 Athena/Atlas 提示符。"""
        output = ""
        tail = ""
        while True:
            word = process.stdout.read(1)
            if not word:
                continue
            output += word
            tail = (tail + word)[-16:]
            if output_state:
                sys.stdout.write(word)
                sys.stdout.flush()
            if tail.endswith("ATHENA>") or tail.endswith("ATLAS>"):
                break
            if tail.endswith(">"):
                break
        return output

    def run_athena_command(self, command, output_state=True):
        """
        运行athena命令并获取输出
        
        Args:
            command (str): 要执行的命令
            output_state (bool): 是否打印输出
        
        Returns:
            str: 命令输出结果
        """
        if output_state:
            print(f"执行命令: {command}\n")

        if not command.endswith('\n'):
            command += '\n'
        self.athena_process.stdin.write(command)
        self.athena_process.stdin.flush()

        if command.lower().strip() == 'quit':
            return ''

        return self._read_until_prompt(self.athena_process, output_state=output_state)

    # 修改输出处理逻辑
    def run_atlas_start(self, output_state=True):
        return self._read_until_prompt(self.atlas_process, output_state=output_state)

    def run_atlas_command(self, command, output_state=True):
        """
        运行Atlas命令并获取输出
        
        Args:
            command (str): 要执行的命令
            output_state (bool): 是否打印输出
        
        Returns:
            str: 命令输出结果
        """
        if output_state:
            print(f"执行命令: {command}\n")

        if not command.endswith('\n'):
            command += '\n'
        self.atlas_process.stdin.write(command)
        self.atlas_process.stdin.flush()

        if command.lower().strip() == 'quit':
            return ''

        return self._read_until_prompt(self.atlas_process, output_state=output_state)
    
    def updateprocess(self):
        cwd1=os.getcwd()
        self.atlas_process=subprocess.Popen(self.atlas,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,bufsize=1,cwd=cwd1)
        self.athena_process=subprocess.Popen(
           self.athena,
           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
           text=True, encoding='UTF-8', errors='replace',  # 或 'ignore'
           bufsize=1, cwd=cwd1
       )
         


if __name__ == '__main__':
    tcad_exe = r"D:\silvaco\exe"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    pycode_dir = os.path.dirname(__file__)
    out_dir = os.path.join(project_root, "saomiao")
    os.makedirs(out_dir, exist_ok=True)

    a = pythontcad()
    a.set_tcad(tcad_exe)
    a.tcad = tcadmodel.topcon_n()
    a.updateprocess()
    a.run_athena_command('line x loc = 0.0  spacing=0.1')
    a.run_athena_command('line x loc = 0.1  spacing=0.1')
    a.run_athena_command('line y loc = 0     spacing = 0.002')
    a.run_athena_command('line y loc = 1.2  spacing = 0.002')
    a.run_athena_command('init silicon phosphor resistivity=3 orientation=100')
    a.run_athena_command('deposit oxide thick=0.50 c.boron=1e21')
    a.run_athena_command('method full.cpl compress')
    a.run_athena_command('diffuse time=240 minutes temp=1000 nitro reflow')
    a.run_athena_command('structure outfile=BSG2.str')


    # # a.read_in('Topcon/topcon_n.in')  #读取in文件

    # a.run() 
    # a.updateprocess()
    # a.run_atlas_start(output_state=True)
    a.tcad.nkpath = pycode_dir + os.sep
    a.tcad.template_lib = os.path.join(pycode_dir, "template.lib")
    a.tcad.resist_rear = 0.02
    a.tcad.taun_Si = 0.1
    a.tcad.Nt_top_N = 1e11
    a.tcad.Nt_polySi_top = 1e14
    for val in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]:
        a.tcad.resist_rear = val
        a.tcad.outlog = os.path.join(out_dir, f"Topcon_n_resist_rear_{val}.log")
        a.tcad.outstr = os.path.join(out_dir, f"Topcon_n_resist_rear_{val}.str")
        a.tcad.outcsv = os.path.join(out_dir, f"Topcon_n_resist_rear_{val}.csv")
        a.tcad.updatecommand()
        a.updateprocess()
        for cmd in a.tcad.get_allcommandlist():
            a.run_atlas_command(cmd)
    a.tcad.resist_rear = 0.02
    a.tcad.taun_Si = 0.1
    a.tcad.Nt_top_N = 1e11
    a.tcad.Nt_polySi_top = 1e14
    for val in [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]:
        a.tcad.taun_Si = val
        a.tcad.outlog = os.path.join(out_dir, f"Topcon_n_taun_Si_{val}.log")
        a.tcad.outstr = os.path.join(out_dir, f"Topcon_n_taun_Si_{val}.str")
        a.tcad.outcsv = os.path.join(out_dir, f"Topcon_n_taun_Si_{val}.csv")
        a.tcad.updatecommand()
        a.updateprocess()
        for cmd in a.tcad.get_allcommandlist():
            a.run_atlas_command(cmd)
    a.tcad.resist_rear = 0.02
    a.tcad.taun_Si = 0.1
    a.tcad.Nt_top_N = 1e11
    a.tcad.Nt_polySi_top = 1e14
    for val in [5e11, 2e11, 1e11, 5e10, 2e10, 1e10]:
        a.tcad.Nt_top_N = val
        a.tcad.outlog = os.path.join(out_dir, f"Topcon_n_Nt_top_N_{val}.log")
        a.tcad.outstr = os.path.join(out_dir, f"Topcon_n_Nt_top_N_{val}.str")
        a.tcad.outcsv = os.path.join(out_dir, f"Topcon_n_Nt_top_N_{val}.csv")
        a.tcad.updatecommand()
        a.updateprocess()
        for cmd in a.tcad.get_allcommandlist():
            a.run_atlas_command(cmd)
    a.tcad.resist_rear = 0.02
    a.tcad.taun_Si = 0.1
    a.tcad.Nt_top_N = 1e11
    a.tcad.Nt_polySi_top = 1e14
    for val in [1e13, 5e13, 1e14, 5e14, 1e15, 5e15]:
        a.tcad.Nt_polySi_top = val
        a.tcad.outlog = os.path.join(out_dir, f"Topcon_n_Nt_polySi_top_{val}.log")
        a.tcad.outstr = os.path.join(out_dir, f"Topcon_n_Nt_polySi_top_{val}.str")
        a.tcad.outcsv = os.path.join(out_dir, f"Topcon_n_Nt_polySi_top_{val}.csv")
        a.tcad.updatecommand()
        a.updateprocess()
        for cmd in a.tcad.get_allcommandlist():
            a.run_atlas_command(cmd)
    # a.set_tcad('D:/silvaco/exe')
    # cwd1=os.getcwd()
    # a.atlas_process=subprocess.Popen(a.atlas,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,bufsize=1,cwd=cwd1)
