from tcaddata import *

class perovskite(tcad):
    def __init__(self) -> None:
        super().__init__()
        # 参数设置
        self.w_pitch = 1
        self.t_glass = 0.135
        self.t_ITO_top = 0.15
        self.t_HTL_top = 0.025
        self.t_PSK_top = 0.2
        self.t_ETL_top = 0.02
        self.t_Ag = 0.2

        # HTL-top 属性
        self.er_HTL_top = 18
        self.x_HTL_top = 3.5
        self.Eg_HTL_top = 1.8
        self.Nc_HTL_top = 2.2e18
        self.Nv_HTL_top = 1.8e19
        self.Na_HTL_top = 1e18
        self.mun_HTL_top = 4.5e-2
        self.mup_HTL_top = 4.5e-2
        self.tn_HTL_top = 5e-7
        self.tp_HTL_top = 5e-7

        # PSK-top 属性
        self.er_PSK_top = 10.4
        self.x_PSK_top = 3.82
        self.Eg_PSK_top = 1.22
        self.Nc_PSK_top = 1.145e18
        self.Nv_PSK_top = 1.44e18
        self.Na_PSK_top = 1e15
        self.mun_PSK_top = 270
        self.mup_PSK_top = 270
        self.tn_PSK_top = 1e-6
        self.tp_PSK_top = 1e-6
        self.Brad_PSK_top = 1e-11
        self.Caugn_PSK_top = 1e-28
        self.Caugp_PSK_top = 1e-28

        # ETL-top 属性
        self.er_ETL_top = 18
        self.x_ETL_top = 4.5
        self.Eg_ETL_top = 1.7
        self.Nc_ETL_top = 2.2e18
        self.Nv_ETL_top = 1.8e19
        self.Na_ETL_top = 1e18
        self.mun_ETL_top = 8e-2
        self.mup_ETL_top = 8e-2
        self.tn_ETL_top = 5e-7
        self.tp_ETL_top = 5e-7
        self.outlog='perovskite.log'
        self.outstr='perovskite.str'
        self.outcsv='perovskite.csv'
        self.nkpath='nk/'
        self.vfinal=1.5
        self.vstep=0.02
        self.vinit=0
        self.tcadparams={
            'w_pitch':self.w_pitch,
            't_glass':self.t_glass,
            't_ITO_top':self.t_ITO_top,
            't_HTL_top':self.t_HTL_top,
            't_PSK_top':self.t_PSK_top,
            't_ETL_top':self.t_ETL_top,
            't_Ag':self.t_Ag,
            'er_HTL_top':self.er_HTL_top,
            'x_HTL_top':self.x_HTL_top,
            'Eg_HTL_top':self.Eg_HTL_top,
            'Nc_HTL_top':self.Nc_HTL_top,
            'Nv_HTL_top':self.Nv_HTL_top,
            'Na_HTL_top':self.Na_HTL_top,
            'mun_HTL_top':self.mun_HTL_top,
            'mup_HTL_top':self.mup_HTL_top,
            'tn_HTL_top':self.tn_HTL_top,
            'tp_HTL_top':self.tp_HTL_top,
            'er_PSK_top':self.er_PSK_top,
            'x_PSK_top':self.x_PSK_top,
            'Eg_PSK_top':self.Eg_PSK_top,
            'Nc_PSK_top':self.Nc_PSK_top,
            'Nv_PSK_top':self.Nv_PSK_top,
            'Na_PSK_top':self.Na_PSK_top,
            'mun_PSK_top':self.mun_PSK_top,
            'mup_PSK_top':self.mup_PSK_top,
            'tn_PSK_top':self.tn_PSK_top,
            'tp_PSK_top':self.tp_PSK_top,
            'Brad_PSK_top':self.Brad_PSK_top,
            'Caugn_PSK_top':self.Caugn_PSK_top,
            'Caugp_PSK_top':self.Caugp_PSK_top,
            'er_ETL_top':self.er_ETL_top,
            'x_ETL_top':self.x_ETL_top,
            'Eg_ETL_top':self.Eg_ETL_top,
            'Nc_ETL_top':self.Nc_ETL_top,
            'Nv_ETL_top':self.Nv_ETL_top,
            'Na_ETL_top':self.Na_ETL_top,
            'mun_ETL_top':self.mun_ETL_top,
            'mup_ETL_top':self.mup_ETL_top,
            'tn_ETL_top':self.tn_ETL_top,
            'tp_ETL_top':self.tp_ETL_top,
            'vfinal':self.vfinal,
            'vstep':self.vstep,
            'vinit':self.vinit, 
        }
        self.updatecommand()
    

    def updatecommand(self):
        a=[]
        a.append(getc(c_MESH(),width=1e11))
        a.append(getc(c_X_M(),location=0.0,spacing=0.5))
        a.append(getc(c_X_M(), location=self.w_pitch, spacing=0.5))
        a.append(getc(c_Y_M(), location=-self.t_HTL_top-self.t_PSK_top-self.t_ETL_top, spacing=0.004))
        a.append(getc(c_Y_M(), location=-self.t_PSK_top-self.t_ETL_top, spacing=0.01))
        a.append(getc(c_Y_M(), location=-self.t_ETL_top, spacing=0.005))
        a.append(getc(c_Y_M(), location=0, spacing=0.005))
        a.append(getc(c_REGION(), num=3, y_max=-self.t_PSK_top-self.t_ETL_top, y_min=-self.t_HTL_top-self.t_PSK_top-self.t_ETL_top, user_mat='pedot'))
        a.append(getc(c_REGION(), num=4, y_max=-self.t_ETL_top, y_min=-self.t_PSK_top-self.t_ETL_top, user_material='peroviskite'))
        a.append(getc(c_REGION(), num=5, y_max=0, y_min=-self.t_ETL_top, user_material='C60'))
        a.append(getc(c_ELECTRODE(), num=1, name='cathode', y_max=0, y_min=0, mat='silver'))
        a.append(getc(c_ELECTRODE(), num=2, name='anode', y_max=-self.t_HTL_top-self.t_PSK_top-self.t_ETL_top, y_min=-self.t_HTL_top-self.t_PSK_top-self.t_ETL_top, mat='ITO'))
        a.append(getc(c_DOPING(), uniform=True, conc=self.Na_HTL_top, p_type=True, reg=3))
        a.append(getc(c_DOPING(), uniform=True, conc=self.Na_PSK_top, p_type=True, reg=4))
        a.append(getc(c_DOPING(), uniform=True, conc=self.Na_ETL_top, n_type=True, reg=5))
        a.append(getc(c_MATERIAL(), mat='ITO', index_file=self.nkpath+'ITO_nk.nk'))
        a.append(getc(c_MATERIAL(), mat='pedot', user_group='semiconductor', index_file=self.nkpath+'pedot.nk'))
        a.append(getc(c_MATERIAL(), mat='peroviskite', user_group='semiconductor', index_file=self.nkpath+'peroviskite.nk'))
        a.append(getc(c_MATERIAL(), mat='C60', user_group='semiconductor', index_file=self.nkpath+'C60_nk.nk'))
        a.append(getc(c_MATERIAL(), mat='silver', index_file=self.nkpath+'Ag_nk.nk'))
        a.append(getc(c_MATERIAL(), region=3, mun=self.mun_HTL_top, mup=self.mup_HTL_top, taun0=self.tn_HTL_top, taup0=self.tp_HTL_top, permittivity=self.er_HTL_top, eg300=self.Eg_HTL_top, affinity=self.x_HTL_top, nc=self.Nc_HTL_top, nv300=self.Nv_HTL_top))
        a.append(getc(c_MATERIAL(), region=4, mun=self.mun_PSK_top, mup=self.mup_PSK_top, taun0=self.tn_PSK_top, taup0=self.tp_PSK_top, permittivity=self.er_PSK_top, eg300=self.Eg_PSK_top, affinity=self.x_PSK_top, nc300=self.Nc_PSK_top, nv300=self.Nv_PSK_top, copt=self.Brad_PSK_top))
        a.append(getc(c_MATERIAL(), region=5, mun=self.mun_ETL_top, mup=self.mup_ETL_top, taun0=self.tn_ETL_top, taup0=self.tp_ETL_top, permittivity=self.er_ETL_top, eg300=self.Eg_ETL_top, affinity=self.x_ETL_top, nc=self.Nc_ETL_top, nv300=self.Nv_ETL_top))
        a.append(getc(c_BEAM(), num=1, x_origin=self.w_pitch/2, y_origin=-self.t_HTL_top-self.t_PSK_top-self.t_ETL_top-5.0, angle=90.0, wavel_start=0.3, wavel_end=1.2, wavel_num=81, reflects=1, tr_matrix=True, diffuse=True, AM1_5=True))
        a.append(getc(c_OUTPUT(), photogen=True, con_band=True, val_band=True, band_param=True, QFN=True, QFP=True, QSS=True, charge=True, flowlines=True, e_mobility=True, e_velocity=True, h_mobility=True, h_velocity=True, recomb=True, u_srh=True, u_radiative=True, u_auger=True, opt_intens=True, taurn=True, taurp=True, traps=True, schottky=True, impact=True, permi=True))
        a.append(getc(c_MODELS(), srh=True, auger=True, fermi=True, ni_fermi=True, optr=True, print=True, conmob=True, ccsmob=True, fldmob=True, bgn=True))
        a.append(getc(c_METHOD(), newton=True, itlimit=50, tun_wkb=True))
        a.append(getc(c_SOLVE(), init=True))
        a.append(getc(c_SOLVE(), b1=1e-8))
        a.append(getc(c_SOLVE(), b1=1e-6))
        a.append(getc(c_SOLVE(), b1=1e-4))
        a.append(getc(c_SOLVE(), b1=1e-2))
        a.append(getc(c_SOLVE(), b1=0.1))
        a.append(getc(c_SOLVE(), b1=1))
        a.append(getc(c_SOLVE(), b1=1.11))
        a.append(getc(c_LOG(), outfile=self.outlog, CSVFILE=self.outcsv,j_hole=True, j_electron=True))
        a.append(getc(c_SOLVE(), vanode=self.vinit, name='anode', vstep=self.vstep, vfinal=self.vfinal))
        a.append(getc(c_LOG(), off=True))
        a.append(getc(c_SAVE(), outfile=self.outstr))
        a.append(getc(c_QUIT()))
        self.commandall=a


class topcon_n(tcad):
    def __init__(self) -> None:
        super().__init__()
        # 厚度参数（μm）
        self.cellwd=1
        self.MgF_thk=0.1
        self.IZO_thk=0.056
        self.al2o3_thk=0.007
        self.Si_thk=195
        self.t_SiO2=0.001
        self.t_polySi_rear_P=0.086
        self.t_Ag=0.1
        self.t_polySi_top_N=0.001

        # 隧穿参数
        self.me=0.5
        self.mh=0.5
        self.quan_thk=0.01

        # 接触电阻（Ω·cm^2）
        self.resist_rear=0.01

        # 界面/寿命等
        self.Cap_area=1e-14
        self.Cap_area_defect=1e-17
        self.taun_Si=0.1
        self.taun_polySi_rear_P=3e-4

        # 掺杂
        self.Nd_top=8e19
        self.Nd_rear=5e20

        # 扩散结深（μm）
        self.front_junc=2
        self.rear_junc=1

        # 新界面态密度
        self.Nt_Si_SiOx=1e13
        self.Nt_SiOx_Poly=1e13
        self.Nt_top_N=1e10

        # 体缺陷密度
        self.Nt_polySi_top=1e14
        self.Nt_polySi_rear=1e14

        # 输出/文件路径
        self.nkpath=''
        self.template_lib='template.lib'
        self.outlog='Topcon_1.log'
        self.outstr='Topcon_1.str'
        self.outcsv='topcon_n.csv'

        # 电扫
        self.vinit=0
        self.vstep1=0.1
        self.vfinal1=0.4
        self.vstep2=0.05
        self.vfinal2=0.8

        self.tcadparams={
            'cellwd':self.cellwd,
            'MgF_thk':self.MgF_thk,
            'IZO_thk':self.IZO_thk,
            'al2o3_thk':self.al2o3_thk,
            'Si_thk':self.Si_thk,
            't_SiO2':self.t_SiO2,
            't_polySi_rear_P':self.t_polySi_rear_P,
            't_Ag':self.t_Ag,
            't_polySi_top_N':self.t_polySi_top_N,
            'me':self.me,
            'mh':self.mh,
            'quan_thk':self.quan_thk,
            'resist_rear':self.resist_rear,
            'Cap_area':self.Cap_area,
            'Cap_area_defect':self.Cap_area_defect,
            'taun_Si':self.taun_Si,
            'taun_polySi_rear_P':self.taun_polySi_rear_P,
            'Nd_top':self.Nd_top,
            'Nd_rear':self.Nd_rear,
            'front_junc':self.front_junc,
            'rear_junc':self.rear_junc,
            'Nt_Si_SiOx':self.Nt_Si_SiOx,
            'Nt_SiOx_Poly':self.Nt_SiOx_Poly,
            'Nt_top_N':self.Nt_top_N,
            'Nt_polySi_top':self.Nt_polySi_top,
            'Nt_polySi_rear':self.Nt_polySi_rear,
            'outlog':self.outlog,
            'outstr':self.outstr,
            'outcsv':self.outcsv,
        }
        self.updatecommand()

    def updatecommand(self):
        a=[]
        # 网格
        a.append(getc(c_MESH(), width=1))
        a.append(getc(c_X_M(), location=0.0, spacing=0.1))
        a.append(getc(c_X_M(), location=self.cellwd, spacing=0.1))

        a.append(getc(c_Y_M(), location=-(self.MgF_thk+self.IZO_thk+self.al2o3_thk), spacing=0.01))
        a.append(getc(c_Y_M(), location=-(self.IZO_thk+self.al2o3_thk), spacing=0.01))
        a.append(getc(c_Y_M(), location=-self.al2o3_thk, spacing=0.005))
        a.append(getc(c_Y_M(), location=0, spacing=0.0005))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+0.005, spacing=0.0005))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+0.05, spacing=0.005))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+0.1, spacing=0.05))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+1, spacing=0.1))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+10, spacing=1))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+15, spacing=5))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk-10, spacing=0.5))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk-5, spacing=0.3))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk-1, spacing=0.01))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk-0.1, spacing=0.002))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk-0.005, spacing=0.0005))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk, spacing=0.0005))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk+0.005, spacing=0.0005))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk+self.t_SiO2+self.t_polySi_rear_P, spacing=0.002))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk+self.t_SiO2+self.t_polySi_rear_P+self.t_Ag, spacing=0.02))

        # 区域
        a.append(getc(c_REGION(), num=9, y_max=-(self.IZO_thk+self.al2o3_thk), y_min=-(self.MgF_thk+self.IZO_thk+self.al2o3_thk), mat='TiO2'))
        a.append(getc(c_REGION(), num=1, y_max=-self.al2o3_thk, y_min=-(self.IZO_thk+self.al2o3_thk), mat='Si3N4'))
        a.append(getc(c_REGION(), num=2, y_max=0, y_min=-self.al2o3_thk, mat='Al2O3'))
        a.append(getc(c_REGION(), num=3, y_max=self.t_polySi_top_N, y_min=0, mat='Silicon'))
        a.append(getc(c_REGION(), num=5, y_max=self.t_polySi_top_N+self.Si_thk, y_min=self.t_polySi_top_N, mat='Silicon'))
        a.append(getc(c_REGION(), num=6, y_max=self.t_polySi_top_N+self.Si_thk+self.t_SiO2, y_min=self.t_polySi_top_N+self.Si_thk, mat='SiO2'))
        a.append(getc(c_REGION(), num=7, y_max=self.t_polySi_top_N+self.Si_thk+self.t_SiO2+self.t_polySi_rear_P, y_min=self.t_polySi_top_N+self.Si_thk+self.t_SiO2, mat='Poly'))
        a.append(getc(c_REGION(), num=8, y_max=self.t_polySi_top_N+self.Si_thk+self.t_SiO2+self.t_polySi_rear_P+self.t_Ag, y_min=self.t_polySi_top_N+self.Si_thk+self.t_SiO2+self.t_polySi_rear_P, mat='Silver'))

        # 电极
        a.append(getc(c_ELECTRODE(), name='cathode', x_min=0, x_max=self.cellwd, y_min=0, y_max=0))
        a.append(getc(c_ELECTRODE(), name='anode', material='silver', region=8))

        # 量子隧穿区域（后侧）
        a.append(getc(c_QTREGION(), x1=0, y1=self.t_polySi_top_N+self.Si_thk-self.quan_thk,
                      x2=self.cellwd, y2=self.t_polySi_top_N+self.Si_thk-self.quan_thk,
                      x3=self.cellwd, y3=self.t_polySi_top_N+self.Si_thk+self.t_SiO2+self.quan_thk,
                      x4=0, y4=self.t_polySi_top_N+self.Si_thk+self.t_SiO2+self.quan_thk,
                      number=1, pts_normal=3, pts_tunnel=100))

        # 掺杂
        a.append(getc(c_DOPING(), uniform=True, conc=self.Nd_top, n_type=True, reg=3))
        a.append(getc(c_DOPING(), uniform=True, conc=2e15, n_type=True, reg=5))
        a.append(getc(c_DOPING(), reg=5, gauss=True, n_type=True, conc=self.Nd_top,
                      peak=self.t_polySi_top_N+0.01, junction=self.front_junc, char=0.1*self.front_junc,
                      x_left=0, x_right=self.cellwd))
        a.append(getc(c_DOPING(), reg=5, gauss=True, p_type=True, conc=self.Nd_rear,
                      peak=self.t_polySi_top_N+self.Si_thk-0.01, junction=self.rear_junc, char=0.1*self.rear_junc,
                      x_left=0, x_right=self.cellwd))
        a.append(getc(c_DOPING(), uniform=True, conc=self.Nd_rear, p_type=True, reg=7))

        # 光学材料（nk）
        a.append(getc(c_MATERIAL(), mat='TiO2', index_file=self.nkpath+'MgF2_nk.nk'))
        a.append(getc(c_MATERIAL(), mat='Si3N4', index_file=self.nkpath+'ITO_nk.nk'))
        a.append(getc(c_MATERIAL(), mat='Silicon', index_file=self.nkpath+'Si_nk.nk'))
        a.append(getc(c_MATERIAL(), mat='SiO2', imag_index=0, real_index=1.5))
        a.append(getc(c_MATERIAL(), mat='Silver', index_file=self.nkpath+'Ag_nk.nk'))
        a.append(getc(c_MATERIAL(), mat='Al2O3', index_file=self.nkpath+'Al2O3_nk.nk'))
        a.append(getc(c_MATERIAL(), mat='Poly', index_file=self.nkpath+'polySi_nk.nk'))

        # 电学材料参数
        a.append(getc(c_MATERIAL(), mat='SiO2', me_tunnel=self.me, mh_tunnel=self.mh, eg300=8.9, affinity=1, permittivity=3.8))
        a.append(getc(c_MATERIAL(), region=5, mun=1000, mup=800, taun0=self.taun_Si, taup0=self.taun_Si, permittivity=11.7, copt=9.5e-15, eg300=1.12, affinity=4.05))
        a.append(getc(c_MATERIAL(), region=3, mun=1000, mup=800, taun0=self.taun_Si, taup0=self.taun_Si, permittivity=11.7, copt=9.5e-15, eg300=1.14, affinity=4.05))
        a.append(getc(c_MATERIAL(), region=7, mun=120, mup=40, taun0=5e-1, taup0=self.taun_polySi_rear_P, permittivity=11.7, copt=9.5e-15, eg300=1.10, affinity=4.05))

        # 体缺陷（前侧扩散、后侧扩散）
        a.append(getc(c_DEFECTS(), F_DEFECTS=self.template_lib, y_max=self.front_junc, y_min=0,
                      NTA=self.Nt_polySi_top, NTD=self.Nt_polySi_top, WTA=0.05, WTD=0.05,
                      NGA=1e16, NGD=1e16, ega=0.56, egd=0.56, wga=0.01, wgd=0.01,
                      SIGGAE=1e-17, SIGGAH=1e-17, SIGGDE=1e-17, SIGGDH=1e-17,
                      SIGTAE=1e-17, SIGTAH=1e-17, SIGTDE=1e-17, SIGTDH=1e-17,
                      continuous=True))
        a.append(getc(c_DEFECTS(), F_DEFECTS=self.template_lib, y_max=self.t_polySi_top_N+self.Si_thk, y_min=self.t_polySi_top_N+self.Si_thk-self.rear_junc,
                      NTA=self.Nt_polySi_rear, NTD=self.Nt_polySi_rear, WTA=0.05, WTD=0.05,
                      NGA=1e16, NGD=1e16, ega=0.56, egd=0.56, wga=0.01, wgd=0.01,
                      SIGGAE=1e-17, SIGGAH=1e-17, SIGGDE=1e-17, SIGGDH=1e-17,
                      SIGTAE=1e-17, SIGTAH=1e-17, SIGTDE=1e-17, SIGTDH=1e-17,
                      continuous=True))

        # 界面缺陷：c‑Si/SiOx、SiOx/Poly、Al2O3/c‑Si
        a.append(getc(c_DEFECTS(), y_max=self.t_polySi_top_N+self.Si_thk, y_min=self.t_polySi_top_N+self.Si_thk-0.005,
                      NTA=5, NTD=5, WTA=1e3, WTD=1e3, NGA=self.Nt_Si_SiOx, NGD=self.Nt_Si_SiOx,
                      ega=0.56, egd=0.56, wga=1e4, wgd=1e4,
                      SIGGAE=self.Cap_area, SIGGAH=self.Cap_area, SIGGDE=self.Cap_area, SIGGDH=self.Cap_area,
                      SIGTAE=self.Cap_area, SIGTAH=self.Cap_area, SIGTDE=self.Cap_area, SIGTDH=self.Cap_area,
                      continuous=True))
        a.append(getc(c_DEFECTS(), y_max=self.t_polySi_top_N+self.Si_thk+self.t_SiO2+0.005, y_min=self.t_polySi_top_N+self.Si_thk+self.t_SiO2,
                      NTA=5, NTD=5, WTA=1e3, WTD=1e3, NGA=self.Nt_SiOx_Poly, NGD=self.Nt_SiOx_Poly,
                      ega=0.56, egd=0.56, wga=1e4, wgd=1e4,
                      SIGGAE=self.Cap_area, SIGGAH=self.Cap_area, SIGGDE=self.Cap_area, SIGGDH=self.Cap_area,
                      SIGTAE=self.Cap_area, SIGTAH=self.Cap_area, SIGTDE=self.Cap_area, SIGTDH=self.Cap_area,
                      continuous=True))
        a.append(getc(c_DEFECTS(), y_max=0.005, y_min=0,
                      NTA=5, NTD=5, WTA=1e3, WTD=1e3, NGA=self.Nt_top_N, NGD=self.Nt_top_N,
                      ega=0.56, egd=0.56, wga=1e4, wgd=1e4,
                      SIGGAE=self.Cap_area, SIGGAH=self.Cap_area, SIGGDE=self.Cap_area, SIGGDH=self.Cap_area,
                      SIGTAE=self.Cap_area, SIGTAH=self.Cap_area, SIGTDE=self.Cap_area, SIGTDH=self.Cap_area,
                      continuous=True))

        # 接触
        a.append(getc(c_CONTACT(), name='cathode', con_resist=self.resist_rear))

        # 结构保存（初始）
        a.append(getc(c_SAVE(), outfile='topcon_0.str'))

        # 光束
        a.append(getc(c_BEAM(), num=1, x_origin=self.cellwd/2, y_origin=-(self.MgF_thk+self.IZO_thk+self.al2o3_thk+10.0),
                      angle=90.0, wavel_start=0.3, wavel_end=1.2, wavel_num=81, reflects=1, tr_matrix=True, diffuse=True, AM1_5=True))

        # 探针
        a.append(getc(c_PROBE(), name='inten', beam=1, intensity=True, max=True))

        # 输出
        a.append(getc(c_OUTPUT(), E_FIELD=True, photogen=True, con_band=True, val_band=True, band_param=True,
                      QFN=True, QFP=True, QSS=True, charge=True, flowlines=True,
                      e_mobility=True, e_velocity=True, h_mobility=True, h_velocity=True,
                      recomb=True, u_srh=True, u_radiative=True, u_auger=True,
                      opt_intens=True, taurn=True, taurp=True, traps=True, schottky=True, impact=True, permi=True))

        # 物理模型
        a.append(getc(c_MODELS(), srh=True, auger=True, fermi=True, ni_fermi=True, optr=True, print=True,
                      conmob=True, ccsmob=True, fldmob=True, bgn=True))
        a.append(getc(c_MODELS(), sis_el=True, sis_ho=True, sis_nlderivs=True, qtregion=1))

        # 数值方法
        a.append(getc(c_METHOD(), newton=True, itlimit=50, tun_wkb=True))

        # 求解流程
        a.append(getc(c_SOLVE(), init=True))
        a.append(getc(c_SOLVE(), b1=1))
        a.append(getc(c_LOG(), outfile=self.outlog, CSVFILE=self.outcsv, j_hole=True, j_electron=True))
        a.append(getc(c_SOLVE(), vanode=self.vinit, name='anode', vstep=self.vstep1, vfinal=self.vfinal1))
        a.append(getc(c_SOLVE(), name='anode', vstep=self.vstep2, vfinal=self.vfinal2))
        a.append(getc(c_LOG(), off=True))
        a.append(getc(c_SAVE(), outfile=self.outstr))
        a.append(getc(c_QUIT()))

        self.commandall=a


class topcon_n_cd(tcad): #自定义掺杂曲线模型
    def __init__(self) -> None:
        super().__init__()
        # 是否保存结构文件
        self.save_str=False
        # 厚度参数（μm）
        self.cellwd=1
        self.MgF_thk=0.1
        self.IZO_thk=0.056
        self.al2o3_thk=0.007
        self.Si_thk=195
        self.t_SiO2=0.001
        self.t_polySi_rear_P=0.086
        self.t_Ag=0.1
        self.t_polySi_top_N=0.001

        # 隧穿参数
        self.me=0.5
        self.mh=0.5
        self.quan_thk=0.01

        # 接触电阻（Ω·cm^2）
        self.resist_rear=0.01

        # 界面/寿命等
        self.Cap_area=1e-14
        self.Cap_area_defect=1e-17
        self.taun_Si=0.1
        self.taun_polySi_rear_P=3e-4

        # 掺杂
        self.Nd_top=8e19
        self.Nd_rear=5e20

        # 扩散结深（μm）
        self.front_junc=2
        self.rear_junc=1

        # 新界面态密度
        self.Nt_Si_SiOx=1e13
        self.Nt_SiOx_Poly=1e13
        self.Nt_top_N=1e10

        # 体缺陷密度
        self.Nt_polySi_top=1e14
        self.Nt_polySi_rear=1e14

        # 输出/文件路径
        self.nkpath=''
        self.cdpath=''
        # 缺陷输入模式：
        # - "mapped": 使用现有 DEFECTS 参数映射法（默认）
        # - "trap_curve": 使用 DOPING TRAP 从外部曲线导入前表面缺陷
        self.defect_mode='mapped'
        self.trap_cdpath=''
        # DOPING TRAP 所需的陷阱寿命参数（电子/空穴都要给）
        self.trap_taun=1e-6
        self.trap_taup=1e-6
        self.template_lib='template.lib'
        self.outlog='Topcon_1.log'
        self.outstr='Topcon_1.str'
        self.outcsv='topcon_n.csv'

        # 电扫
        self.vinit=0
        self.vstep1=0.1
        self.vfinal1=0.4
        self.vstep2=0.05
        self.vfinal2=0.8

        self.tcadparams={
            'cellwd':self.cellwd,
            'MgF_thk':self.MgF_thk,
            'IZO_thk':self.IZO_thk,
            'al2o3_thk':self.al2o3_thk,
            'Si_thk':self.Si_thk,
            't_SiO2':self.t_SiO2,
            't_polySi_rear_P':self.t_polySi_rear_P,
            't_Ag':self.t_Ag,
            't_polySi_top_N':self.t_polySi_top_N,
            'me':self.me,
            'mh':self.mh,
            'quan_thk':self.quan_thk,
            'resist_rear':self.resist_rear,
            'Cap_area':self.Cap_area,
            'Cap_area_defect':self.Cap_area_defect,
            'taun_Si':self.taun_Si,
            'taun_polySi_rear_P':self.taun_polySi_rear_P,
            'Nd_top':self.Nd_top,
            'Nd_rear':self.Nd_rear,
            'front_junc':self.front_junc,
            'rear_junc':self.rear_junc,
            'Nt_Si_SiOx':self.Nt_Si_SiOx,
            'Nt_SiOx_Poly':self.Nt_SiOx_Poly,
            'Nt_top_N':self.Nt_top_N,
            'Nt_polySi_top':self.Nt_polySi_top,
            'Nt_polySi_rear':self.Nt_polySi_rear,
            'defect_mode':self.defect_mode,
            'trap_cdpath':self.trap_cdpath,
            'trap_taun':self.trap_taun,
            'trap_taup':self.trap_taup,
            'outlog':self.outlog,
            'outstr':self.outstr,
            'outcsv':self.outcsv,
        }
        self.updatecommand()

    def updatecommand(self):
        a=[]
        # 网格
        a.append(getc(c_MESH(), width=1))
        a.append(getc(c_X_M(), location=0.0, spacing=0.1))
        a.append(getc(c_X_M(), location=self.cellwd, spacing=0.1))

        a.append(getc(c_Y_M(), location=-(self.MgF_thk+self.IZO_thk+self.al2o3_thk), spacing=0.01))
        a.append(getc(c_Y_M(), location=-(self.IZO_thk+self.al2o3_thk), spacing=0.01))
        a.append(getc(c_Y_M(), location=-self.al2o3_thk, spacing=0.005))
        a.append(getc(c_Y_M(), location=0, spacing=0.0005))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+0.005, spacing=0.0005))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+0.05, spacing=0.005))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+0.1, spacing=0.05))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+1, spacing=0.1))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+10, spacing=1))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+15, spacing=5))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk-10, spacing=0.5))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk-5, spacing=0.3))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk-1, spacing=0.01))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk-0.1, spacing=0.002))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk-0.005, spacing=0.0005))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk, spacing=0.0005))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk+0.005, spacing=0.0005))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk+self.t_SiO2+self.t_polySi_rear_P, spacing=0.002))
        a.append(getc(c_Y_M(), location=self.t_polySi_top_N+self.Si_thk+self.t_SiO2+self.t_polySi_rear_P+self.t_Ag, spacing=0.02))

        # 区域
        a.append(getc(c_REGION(), num=9, y_max=-(self.IZO_thk+self.al2o3_thk), y_min=-(self.MgF_thk+self.IZO_thk+self.al2o3_thk), mat='TiO2'))
        a.append(getc(c_REGION(), num=1, y_max=-self.al2o3_thk, y_min=-(self.IZO_thk+self.al2o3_thk), mat='Si3N4'))
        a.append(getc(c_REGION(), num=2, y_max=0, y_min=-self.al2o3_thk, mat='Al2O3'))
        a.append(getc(c_REGION(), num=3, y_max=self.t_polySi_top_N, y_min=0, mat='Silicon'))
        a.append(getc(c_REGION(), num=5, y_max=self.t_polySi_top_N+self.Si_thk, y_min=self.t_polySi_top_N, mat='Silicon'))
        a.append(getc(c_REGION(), num=6, y_max=self.t_polySi_top_N+self.Si_thk+self.t_SiO2, y_min=self.t_polySi_top_N+self.Si_thk, mat='SiO2'))
        a.append(getc(c_REGION(), num=7, y_max=self.t_polySi_top_N+self.Si_thk+self.t_SiO2+self.t_polySi_rear_P, y_min=self.t_polySi_top_N+self.Si_thk+self.t_SiO2, mat='Poly'))
        a.append(getc(c_REGION(), num=8, y_max=self.t_polySi_top_N+self.Si_thk+self.t_SiO2+self.t_polySi_rear_P+self.t_Ag, y_min=self.t_polySi_top_N+self.Si_thk+self.t_SiO2+self.t_polySi_rear_P, mat='Silver'))

        # 电极
        a.append(getc(c_ELECTRODE(), name='cathode', x_min=0, x_max=self.cellwd, y_min=0, y_max=0))
        a.append(getc(c_ELECTRODE(), name='anode', material='silver', region=8))

        # 量子隧穿区域（后侧）
        a.append(getc(c_QTREGION(), x1=0, y1=self.t_polySi_top_N+self.Si_thk-self.quan_thk,
                      x2=self.cellwd, y2=self.t_polySi_top_N+self.Si_thk-self.quan_thk,
                      x3=self.cellwd, y3=self.t_polySi_top_N+self.Si_thk+self.t_SiO2+self.quan_thk,
                      x4=0, y4=self.t_polySi_top_N+self.Si_thk+self.t_SiO2+self.quan_thk,
                      number=1, pts_normal=3, pts_tunnel=100))

        # 掺杂
        a.append(getc(c_DOPING(), uniform=True, conc=self.Nd_top, n_type=True, reg=3))
        a.append(getc(c_DOPING(), uniform=True, conc=2e15, n_type=True, reg=5))
        a.append(getc(c_DOPING(), reg=5, ASCII=True, infile=self.cdpath, n_type=True))
        # trap_curve 模式：将外部缺陷曲线作为 TRAP 态密度导入前部区域
        if self.defect_mode == 'trap_curve' and self.trap_cdpath:
            a.append(
                getc(
                    c_DOPING(),
                    reg=5,
                    ASCII=True,
                    infile=self.trap_cdpath,
                    n_type=True,
                    trap=True,
                    taun=self.trap_taun,
                    taup=self.trap_taup,
                )
            )
        a.append(getc(c_DOPING(), reg=5, gauss=True, p_type=True, conc=self.Nd_rear,
                      peak=self.t_polySi_top_N+self.Si_thk-0.01, junction=self.rear_junc, char=0.1*self.rear_junc,
                      x_left=0, x_right=self.cellwd))
        a.append(getc(c_DOPING(), uniform=True, conc=self.Nd_rear, p_type=True, reg=7))

        # 光学材料（nk）
        a.append(getc(c_MATERIAL(), mat='TiO2', index_file=self.nkpath+'MgF2_nk.nk'))
        a.append(getc(c_MATERIAL(), mat='Si3N4', index_file=self.nkpath+'ITO_nk.nk'))
        a.append(getc(c_MATERIAL(), mat='Silicon', index_file=self.nkpath+'Si_nk.nk'))
        a.append(getc(c_MATERIAL(), mat='SiO2', imag_index=0, real_index=1.5))
        a.append(getc(c_MATERIAL(), mat='Silver', index_file=self.nkpath+'Ag_nk.nk'))
        a.append(getc(c_MATERIAL(), mat='Al2O3', index_file=self.nkpath+'Al2O3_nk.nk'))
        a.append(getc(c_MATERIAL(), mat='Poly', index_file=self.nkpath+'polySi_nk.nk'))

        # 电学材料参数
        a.append(getc(c_MATERIAL(), mat='SiO2', me_tunnel=self.me, mh_tunnel=self.mh, eg300=8.9, affinity=1, permittivity=3.8))
        a.append(getc(c_MATERIAL(), region=5, mun=1000, mup=800, taun0=self.taun_Si, taup0=self.taun_Si, permittivity=11.7, copt=9.5e-15, eg300=1.12, affinity=4.05))
        a.append(getc(c_MATERIAL(), region=3, mun=1000, mup=800, taun0=self.taun_Si, taup0=self.taun_Si, permittivity=11.7, copt=9.5e-15, eg300=1.14, affinity=4.05))
        a.append(getc(c_MATERIAL(), region=7, mun=120, mup=40, taun0=5e-1, taup0=self.taun_polySi_rear_P, permittivity=11.7, copt=9.5e-15, eg300=1.10, affinity=4.05))

        # 体缺陷：前侧根据模式切换；后侧保持原模型
        if self.defect_mode != 'trap_curve':
            a.append(getc(c_DEFECTS(), F_DEFECTS=self.template_lib, y_max=self.front_junc, y_min=0,
                          NTA=self.Nt_polySi_top, NTD=self.Nt_polySi_top, WTA=0.05, WTD=0.05,
                          NGA=1e16, NGD=1e16, ega=0.56, egd=0.56, wga=0.01, wgd=0.01,
                          SIGGAE=1e-17, SIGGAH=1e-17, SIGGDE=1e-17, SIGGDH=1e-17,
                          SIGTAE=1e-17, SIGTAH=1e-17, SIGTDE=1e-17, SIGTDH=1e-17,
                          continuous=True))
        a.append(getc(c_DEFECTS(), F_DEFECTS=self.template_lib, y_max=self.t_polySi_top_N+self.Si_thk, y_min=self.t_polySi_top_N+self.Si_thk-self.rear_junc,
                      NTA=self.Nt_polySi_rear, NTD=self.Nt_polySi_rear, WTA=0.05, WTD=0.05,
                      NGA=1e16, NGD=1e16, ega=0.56, egd=0.56, wga=0.01, wgd=0.01,
                      SIGGAE=1e-17, SIGGAH=1e-17, SIGGDE=1e-17, SIGGDH=1e-17,
                      SIGTAE=1e-17, SIGTAH=1e-17, SIGTDE=1e-17, SIGTDH=1e-17,
                      continuous=True))

        # 界面缺陷：c‑Si/SiOx、SiOx/Poly、Al2O3/c‑Si
        a.append(getc(c_DEFECTS(), y_max=self.t_polySi_top_N+self.Si_thk, y_min=self.t_polySi_top_N+self.Si_thk-0.005,
                      NTA=5, NTD=5, WTA=1e3, WTD=1e3, NGA=self.Nt_Si_SiOx, NGD=self.Nt_Si_SiOx,
                      ega=0.56, egd=0.56, wga=1e4, wgd=1e4,
                      SIGGAE=self.Cap_area, SIGGAH=self.Cap_area, SIGGDE=self.Cap_area, SIGGDH=self.Cap_area,
                      SIGTAE=self.Cap_area, SIGTAH=self.Cap_area, SIGTDE=self.Cap_area, SIGTDH=self.Cap_area,
                      continuous=True))
        a.append(getc(c_DEFECTS(), y_max=self.t_polySi_top_N+self.Si_thk+self.t_SiO2+0.005, y_min=self.t_polySi_top_N+self.Si_thk+self.t_SiO2,
                      NTA=5, NTD=5, WTA=1e3, WTD=1e3, NGA=self.Nt_SiOx_Poly, NGD=self.Nt_SiOx_Poly,
                      ega=0.56, egd=0.56, wga=1e4, wgd=1e4,
                      SIGGAE=self.Cap_area, SIGGAH=self.Cap_area, SIGGDE=self.Cap_area, SIGGDH=self.Cap_area,
                      SIGTAE=self.Cap_area, SIGTAH=self.Cap_area, SIGTDE=self.Cap_area, SIGTDH=self.Cap_area,
                      continuous=True))
        if self.defect_mode != 'trap_curve':
            a.append(getc(c_DEFECTS(), y_max=0.005, y_min=0,
                          NTA=5, NTD=5, WTA=1e3, WTD=1e3, NGA=self.Nt_top_N, NGD=self.Nt_top_N,
                          ega=0.56, egd=0.56, wga=1e4, wgd=1e4,
                          SIGGAE=self.Cap_area, SIGGAH=self.Cap_area, SIGGDE=self.Cap_area, SIGGDH=self.Cap_area,
                          SIGTAE=self.Cap_area, SIGTAH=self.Cap_area, SIGTDE=self.Cap_area, SIGTDH=self.Cap_area,
                          continuous=True))

        # 接触
        a.append(getc(c_CONTACT(), name='cathode', con_resist=self.resist_rear))

        # 结构保存（初始）
        a.append(getc(c_SAVE(), outfile='topcon_0.str'))

        # 光束
        a.append(getc(c_BEAM(), num=1, x_origin=self.cellwd/2, y_origin=-(self.MgF_thk+self.IZO_thk+self.al2o3_thk+10.0),
                      angle=90.0, wavel_start=0.3, wavel_end=1.2, wavel_num=81, reflects=1, tr_matrix=True, diffuse=True, AM1_5=True))

        # 探针
        a.append(getc(c_PROBE(), name='inten', beam=1, intensity=True, max=True))

        # 输出
        a.append(getc(c_OUTPUT(), E_FIELD=True, photogen=True, con_band=True, val_band=True, band_param=True,
                      QFN=True, QFP=True, QSS=True, charge=True, flowlines=True,
                      e_mobility=True, e_velocity=True, h_mobility=True, h_velocity=True,
                      recomb=True, u_srh=True, u_radiative=True, u_auger=True,
                      opt_intens=True, taurn=True, taurp=True, traps=True, schottky=True, impact=True, permi=True))

        # 物理模型
        a.append(getc(c_MODELS(), srh=True, auger=True, fermi=True, ni_fermi=True, optr=True, print=True,
                      conmob=True, ccsmob=True, fldmob=True, bgn=True))
        a.append(getc(c_MODELS(), sis_el=True, sis_ho=True, sis_nlderivs=True, qtregion=1))

        # 数值方法
        a.append(getc(c_METHOD(), newton=True, itlimit=50, tun_wkb=True))

        # 求解流程
        a.append(getc(c_SOLVE(), init=True))
        a.append(getc(c_SOLVE(), b1=1))
        a.append(getc(c_LOG(), outfile=self.outlog, CSVFILE=self.outcsv, j_hole=True, j_electron=True))
        a.append(getc(c_SOLVE(), vanode=self.vinit, name='anode', vstep=self.vstep1, vfinal=self.vfinal1))
        a.append(getc(c_SOLVE(), name='anode', vstep=self.vstep2, vfinal=self.vfinal2))
        a.append(getc(c_LOG(), off=True))
        if self.save_str:
            a.append(getc(c_SAVE(), outfile=self.outstr))
        else:
            a.append(getc(c_SAVE(), outfile='topcon_0.str'))
        a.append(getc(c_QUIT()))

        self.commandall=a
if __name__ == '__main__':
    tcad1=perovskite()
    print(tcad1.getparams())
    # a=getc(c_DOPING(), uniform=True, conc=1e15, p_type=True, reg=3)
    # print(a.get_command())