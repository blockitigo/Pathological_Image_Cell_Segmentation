import streamlit as st
import os
from streamlit_option_menu import option_menu

import pandas as pd
from raceplotly.plots import barplot
from collections import deque
import json
import train
import test
import tools.analysis_tools.analyze_logs as alog

def train_model(params):
    st.write("开始训练模型...")
    st.write("训练参数：", params)
    # 在这里执行模型训练的逻辑
    model=train.run_model(params)
    # st.write(model)
    st.write('Train success!!!!⭐')

# 获取下一级目录中的文件夹，主要用来获取data的根目录
def get_all_folders(folder_path, relative_path=""):
    folders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            relative_item_path = os.path.join(relative_path, item) + "/"
            folders.append(relative_item_path)
    return folders

# 上传，获取配置文件
def creat_upfile_config():
    config_file = st.file_uploader("上传一张测试图片", type=["jpg","png"])
    config_path=''
    filename=''
    if config_file is not None:
        print(config_file)
        config_filename = config_file.name  # 获取上传文件的文件名
        filename=config_filename
        config_path = f"./uploaded_files/{config_filename}"  # 指定保存文件的路径
        with open(config_path, "wb") as f:
            f.write(config_file.getvalue())  # 保存上传文件到指定路径
        st.write(config_path)
    return config_path,filename

# 获取指定文件夹下的文件
def list_files_in_directory(directory):
    files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):  # 仅添加文件，排除目录
            files.append(file)
    return files

# 获取类别名字
def get_multiple_names():
    names = []
    num_names = st.sidebar.number_input("请输入名字数量", min_value=1, step=1)
    for i in range(num_names):
        name = st.sidebar.text_input(f"请输入名字 {i+1}")
        names.append(name)
    return names

# 生成种类颜色
def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        red = (i * 71) % 256
        green = (i * 113) % 256
        blue = (i * 157) % 256
        colors.append((red, green, blue))
    return colors

def creat_page_train():
    sbdir = st.sidebar.form("训练参数")
        # 创建侧边栏来设置训练参数
    sbdir.title("训练参数设置")
    # 设置轮数、学习率、批次大小
    epochs = sbdir.slider("训练轮数", min_value=1, max_value=10, value=5)
    learning_rate = sbdir.slider("学习率", min_value=0.001, max_value=0.02, value=0.02)
    batch_size = sbdir.selectbox("批次大小", options=[1, 2, 4, 8,16], index=2)

    # 过滤出文件夹
    # folders = get_all_folders('./data/', './data/')
    # 选取数据集
    # data_root = st.sidebar.selectbox("数据根目录", folders)

    # 选择配置文件
    config_model='./config_model/'
    config_path=config_model+sbdir.selectbox("请选择一个配置文件", list_files_in_directory(config_model))

    # 输入类别数目
    # names = get_multiple_names()
    # metainfo={'classes': ('',), 'palette': [(0, 0, 0)]}
    # metainfo['classes'] = tuple(names)
    # metainfo['palette'] = generate_colors(len(names))

    if sbdir.form_submit_button("测试参数"):
        train_params = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            # "data_folder": data_root,
            "config": config_path,
            # "metainfo": metainfo
        }
        st.write(train_params)
    # 创建开始训练按钮
    if sbdir.form_submit_button("开始训练"):
        train_params = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            # "data_root": data_root,
            "config": config_path,
            # "metainfo": metainfo
        }
        train_model(train_params)

def creat_page_test():
    sbdir = st.sidebar.form("训练参数")
    sbdir.title("训练参数")
    # 选择配置文件
    config_model='./config_model/'
    config_path=config_model+sbdir.selectbox("请选择一个配置文件", list_files_in_directory(config_model))
    epoch='./work_dirs/'
    epoch_path=epoch+sbdir.selectbox("请选择一个权重文件", list_files_in_directory(epoch))
    img_path,filename=creat_upfile_config()
    
    if sbdir.form_submit_button("测试参数"):
        train_params = {
            "epochs": epoch_path,
            "config_path": config_path,
            "img_path": img_path,
            "filename": filename

        }
        st.write(train_params)
    if sbdir.form_submit_button("预测"):
        train_params = {
            "epochs": epoch_path,
            "config_path": config_path,
            "img_path": img_path,
            "filename": filename
        }
        res_path=test.predict(train_params)
        # 创建Streamlit的两列布局
        col1, col2 = st.columns(2)
        # 在第一列显示第一张图片
        col1.image(img_path, use_column_width=True)

        # 在第二列显示第二张图片
        col2.image(res_path, use_column_width=True)

def creat_page_view():
    st.markdown(""" <style> .font {
    font-size:25px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">上传文件...</p>', unsafe_allow_html=True) #use st.markdown
    uploaded_file = st.file_uploader('', type=["json"])
    if uploaded_file is not None:    
        # 读取上传的JSON文件并转换为DataFrame
        data_list = []
        mmap_list=[]
        file_contents = uploaded_file.read()
        file_contents = file_contents.decode('utf-8')
        lines = file_contents.split('\n')
        for line in lines:
            if line.strip():
                data = json.loads(line)
                if list(data.keys())[0]!='lr':
                    mmap_list.append(data)
                else:
                    data_list.append(data)
        df = pd.DataFrame(data_list) 
        mmap_df=pd.DataFrame(mmap_list)
        st.write(df)
        st.write(mmap_df)
        # 动画
        st.write('---')
        st.markdown('<p class="font">设置参数...</p>', unsafe_allow_html=True)
        column_list=list(df)
        column_list = deque(column_list)
        column_list.appendleft('-')
        df.insert(0, '数值', '数值')
        with st.form(key='columns_in_form'):
            text_style = '<p style="font-family:sans-serif; color:red; font-size: 15px;">***下面两列是必填项***</p>'
            st.markdown(text_style, unsafe_allow_html=True)
            col2, col3 = st.columns( [ 1, 1])
            # with col1:
            item_column='数值'
            #     st.write('you choose item_column:',item_column)
            with col2:    
                value_column=st.selectbox('应变量:',column_list, index=0, help='希望观察哪个数据的变化') 
            with col3:    
                time_column=st.selectbox('自变量:',column_list, index=0, help='由哪个数据引起的变化，即按照什么序列变化')  

            text_style = '<p style="font-family:sans-serif; color:blue; font-size: 15px;">***微调选项（可选）***</p>'
            st.markdown(text_style, unsafe_allow_html=True)
            col4, col5, col6 = st.columns( [1, 1, 1])
            with col4:
                direction=st.selectbox('选择数据变化方向:',['-','横向变化','纵向变化'], index=0, help='默认横向变化' ) 
                if direction=='横向变化'or direction=='-':
                    orientation='horizontal'
                elif  direction=='纵向变化':   
                    orientation='vertical'
            with col5:
                item_label=st.text_input('添加纵轴标签:')  
            with col6:
                value_label=st.text_input('添加横轴标签')      
            col10, col11, col12 = st.columns( [1, 1, 1])
            with col10:
                speed=st.slider('动画速度',10,500,100, step=10)
                frame_duration=500-speed  
            with col11:
                chart_width=st.slider('表格宽度',500,1000,500, step=20)
            with col12:    
                chart_height=st.slider('表格高度',100,1000,300, step=20)
        
            submitted = st.form_submit_button('提交')
        st.write('---')
        if submitted:        
            if  value_column=='-'or time_column=='-':
                st.warning("请完成两个必填项")
            else: 
                st.markdown('<p class="font">生成图形中... 完成!</p>', unsafe_allow_html=True)   
                df['time_column'] = pd.to_datetime(df[time_column])
                df['value_column'] = df[value_column].astype(float)
     
                raceplot = barplot(df,  item_column=item_column, value_column=value_column, time_column=time_column)
                fig=raceplot.plot(item_label = item_label, value_label = value_label, time_label = time_column+'s:', frame_duration = frame_duration,orientation=orientation)
                fig.update_layout(
                # title=chart_title,
                autosize=False,
                width=chart_width,
                height=chart_height,
                paper_bgcolor="lightgray",
                )
                st.plotly_chart(fig, use_container_width=True)
                
        # 2222222222222222222222
        st.write('---')
        st.markdown('<p class="font">设置参数...</p>', unsafe_allow_html=True)
        column_list=list(mmap_df)
        column_list = deque(column_list)
        column_list.appendleft('-')
        mmap_df.insert(0, '数值', '数值')
        with st.form(key='columns_in_form2'):
            text_style = '<p style="font-family:sans-serif; color:red; font-size: 15px;">***下面两列是必填项***</p>'
            st.markdown(text_style, unsafe_allow_html=True)
            col2, col3 = st.columns( [ 1, 1])
            # with col1:
            item_column='数值'
            #     st.write('you choose item_column:',item_column)
            with col2:    
                value_column=st.selectbox('应变量:',column_list, index=0, help='希望观察哪个数据的变化') 
            with col3:    
                time_column=st.selectbox('自变量:',column_list, index=0, help='由哪个数据引起的变化，即按照什么序列变化')  

            text_style = '<p style="font-family:sans-serif; color:blue; font-size: 15px;">***微调选项（可选）***</p>'
            st.markdown(text_style, unsafe_allow_html=True)
            col4, col5, col6 = st.columns( [1, 1, 1])
            with col4:
                direction=st.selectbox('选择数据变化方向:',['-','横向变化','纵向变化'], index=0, help='默认横向变化' ) 
                if direction=='横向变化'or direction=='-':
                    orientation='horizontal'
                elif  direction=='纵向变化':   
                    orientation='vertical'
            with col5:
                item_label=st.text_input('添加纵轴标签:')  
            with col6:
                value_label=st.text_input('添加横轴标签')      
            col10, col11, col12 = st.columns( [1, 1, 1])
            with col10:
                speed=st.slider('动画速度',10,500,100, step=10)
                frame_duration=500-speed  
            with col11:
                chart_width=st.slider('表格宽度',500,1000,500, step=20)
            with col12:    
                chart_height=st.slider('表格高度',100,1000,300, step=20)
        
            submitted = st.form_submit_button('提交')
        st.write('---')
        if submitted:        
            if  value_column=='-'or time_column=='-':
                st.warning("请完成两个必填项")
            else: 
                st.markdown('<p class="font">生成图形中... 完成!</p>', unsafe_allow_html=True)   
                mmap_df['time_column'] = pd.to_datetime(mmap_df[time_column])
                mmap_df['value_column'] = mmap_df[value_column].astype(float)
     
                raceplot = barplot(mmap_df,  item_column=item_column, value_column=value_column, time_column=time_column)
                fig=raceplot.plot(item_label = item_label, value_label = value_label, time_label = time_column+'s:', frame_duration = frame_duration,orientation=orientation)
                fig.update_layout(
                # title=chart_title,
                autosize=False,
                width=chart_width,
                height=chart_height,
                paper_bgcolor="lightgray",
                )
                st.plotly_chart(fig, use_container_width=True)

def creat_page_draw():
    sbdir = st.sidebar.form("diy参数画图")
    sbdir.title("diy参数画图")
    json_path='./json/'+sbdir.selectbox("请选择一个配置文件", list_files_in_directory('./json'))
    keys=sbdir.multiselect("选择纵坐标（可多选）",
                      ["lr", "data_time", "loss", "loss_rpn_cls", "loss_rpn_bbox", "loss_cls", "acc", "loss_bbox", "loss_mask", "loss_mask_iou", "time", "epoch", "memory", "step"],
                      ["loss_bbox", "loss_mask","loss_cls"])
    if sbdir.form_submit_button("开始画图"):
        json_ppth=[]
        json_ppth.append(json_path)
        img_path=alog.draw_view(json_ppth,keys)
        print(img_path)

        st.image(img_path)

def main():
    # 设置页面标题和描述
    with st.sidebar:
        choose = option_menu("Main Menu", ["Train", "Test","View", "Draw"],
                         icons=['house', 'file-slides','app-indicator','person lines fill'],
                         menu_icon="list", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

    st.title("MMDETECTION VIEW")
    st.header("这是一个简单的模型展示示例")
    # navigation = st.sidebar.radio("Navigation", ["Train", "Test"])
    if choose == "Train":
        creat_page_train()
    elif choose == "Test":
        creat_page_test()
    elif choose=="View":
        creat_page_view()
    elif choose=="Draw":
        creat_page_draw()

    # 在这里添加其他的应用程序逻辑

if __name__ == '__main__':
    main()
