import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict
st.set_option("deprecation.showfileUploaderEncoding",False)
st.sidebar.title("岩石標本の打音グラフ画像認識アプリ")#岩石標本の打音グラフ画像認識アプリ
st.sidebar.write("岩石標本の打音データをグラフ化した画像データを学習したモデルで岩石の判定を行います")#岩石標本の打音データをグラフ化した画像データを学習したモデルで岩石の判定を行います

st.sidebar.write("")


#img_source = st.sidebar.radio("画像のソースを選択してください。","画像をアップロード")
#if img_source == "画像をアップロード":

img_file = st.sidebar.file_uploader("画像を選択してください。",type=["png","jpg","jpeg"])

if img_file is not None:
    with st.spinner("判定中"):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        #予測
        results = predict(img)

        #結果の表示
        st.subheader("判定結果")
        n_top = 3#確率が高い順に3つまで返す
        #for result in results[:n_top]:
        for result in results:
            st.write(str(round(result[2]*100,2)) + "%の確率で" + result[0] + "です。")

        n_top = 3

        #円グラフの表示
        pie_labels = [result[1] for result in results]
        #pie_labels = [result[1] for result in results[:n_top]]
        #pie_labels.append("others")#その他
        pie_probs = [result[2] for result in results]
        #pie_probs = [result[2] for result in results[:n_top]]
        #pie_probs.append(sum([result[2] for result in results[n_top:]])) #その他

        # 0％の確率を持つ要素を削除--------------------------------------------------
        # Filter out labels and probabilities with 0% probability
        for result in results:
            filtered_labels = [label for label, prob in zip(pie_labels, pie_probs) if prob >= 0.00001]
            filtered_probs = [prob for prob in pie_probs if prob >= 0.00001]

        #others_prob = sum(prob for prob in pie_probs if prob < 0.0001)
        #if others_prob > 0:
        #     filtered_labels.append("others")
        #     filtered_probs.append(others_prob)

        #---------------------------------------------------------------------------

        fig, ax = plt.subplots()
        wedgeprops={"width":0.3, "edgecolor":"white"}
        textprops = {"fontsize":6}
        ax.pie(filtered_probs, labels=filtered_labels, counterclock=False, startangle=90,textprops=textprops, autopct="%.2f",wedgeprops=wedgeprops) #円グラフ pie_probs, labels=pie_labels
        st.pyplot(fig)

        st.sidebar.write("")
        st.sidebar.write("")
