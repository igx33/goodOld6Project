import math
import cv2
import numpy as np
from sklearn.datasets import fetch_mldata
from skimage.measure import regionprops
from skimage import color
from scipy import ndimage
from skimage.measure import label


# RA 30/2013  -  Igor Mandic

cccounterx=-1
mnist = fetch_mldata('MNIST original', data_home='data_home_thingy')
novi_MNIST_set_podataka=[]


def sledeciID():
    #print("novi broj")

    global cccounterx
    cccounterx+=1
    #print("BR novi je: "+str(cccounterx))
    return cccounterx

def uDosegu(x1,x2,x3):
    vrijed = []
    for obj in x3:
        p1,p2 = x2['center']
        p3,p4 = obj['center']
        uio = (p3 - p1, p4 - p2)
        re1, re2 = uio
        mk = math.sqrt(re1 * re1 + re2 * re2)
        #print("MK: " + str(mk))

        if(mk<x1):
            vrijed.append(obj)

    #print("VRIJEDNOST: " + str(vrijed))
    return vrijed


def tackaDoLinije(x1,x2,x3):
    p1,p2=x2
    p3,p4=x3
    linijski_vektor = (p3-p1,p4-p2)
    p1,p2=x2
    p3,p4=x1
    tacka_vektor= (p3-p1,p4-p2)
    p1,p2 = linijski_vektor
    duzina_linije = math.sqrt(p1*p1 + p2*p2)
    r1,r2 =linijski_vektor
    ww = math.sqrt(r1*r1 + r2*r2)
    jedinicna_linija = (r1/ww,r2/ww)
    p1,p2 = tacka_vektor
    tacka_vektor_skalirana = (p1* (1.0/duzina_linije) ,p2 *(1.0/duzina_linije))
    q1,q2 = jedinicna_linija
    w1,w2 = tacka_vektor_skalirana
    xx= q1*w1+q2*w2

    xxx=1
    if xx<0.0:
        xx=vratiIspravnu(0.0)
        xxx=vratiIspravnu(-1)
    elif xx>1.0:
        xx=vratiIspravnu(1.0)
        xxx=vratiIspravnu(-1)

    p1,p2 = linijski_vektor
    najblizi = (p1*xx,p2*xx)
    p1,p2 = najblizi
    p3,p4 = tacka_vektor
    uio = (p3-p1,p4-p2)
    re1,re2 = uio
    com = math.sqrt(re1*re1 + re2*re2)
    distanca_ova = com
    p1,p2 = najblizi
    p3,p4 = x2
    najblizi = (p1+p3, p2-p4)

    #print("Distanca: " + str(distanca_ova))
    return (distanca_ova, (int(najblizi[0]),int(najblizi[1])),xxx)

def nuller():
    #if x==1:
        #return [0,0]
    #if x==0:
        #return 0
    return 0

def hjuovaTransformacija(frejm,sivaSliba, param):
    ivice = cv2.Canny(sivaSliba,100,120,apertureSize=3)
    #tackice

    m1=nuller()
    m2=nuller()
    m3=nuller()
    m4=nuller()
    linije = cv2.HoughLinesP(ivice,1,np.pi/180,30,700,10)

    #m1,m2,m3,m4 = linije[0][0]
    for q1,q2,q3,q4 in linije[0]:
        m1=q1
        m2=q2
        m3=q3
        m4=q4

    for i in range(len(linije)):
        for z1,z2,z3,z4 in linije[i]:
            if z1<m1:
                m1=z1
                m2=z2
            if z3>m3:
                m3=z3
                m4=z4
    #cv2.line(frejm,(m1,m2),(m3,m4),(0,255,0),param)

    #print("")

   # print("HTRANS: " + str(m1) + " " + str(m2) + " " + str(m3) + " " + str(m4) + " ")

    return m1,m2,m3,m4

def nadjiParametreLinije(nazivVidea):
    cap=cv2.VideoCapture(nazivVidea)

    l=nuller()
    siva = "grayFrame"
    #frejm1="frame"
    if l==0:
        l=l+1
        while(cap.isOpened()):
            ret,frejm=cap.read()
            if ret:
                siva=cv2.cvtColor(frejm,cv2.COLOR_BGR2GRAY)
                #frejm1=frejm
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
    return hjuovaTransformacija(frejm,siva,2)

def vratiIspravnu(zx):
    #sta ako je niz?
    #if type(zx) :

    #if type(zx) :
    return zx

def prepoznajRegione(slika_BW):

    naziv_slike = label(slika_BW)
    regioni = regionprops(naziv_slike)
    #print("jes")
    #slika_GGx = img_as_ubyte(regioni[0])
    #cv2.imshow('slikaStara GG', slika_GGx)
    #cv2.waitKey(2000)
    #print("jes")
    regioni1 = []
    regioniXX = nuller()
    for region in regioni:
        regioniXX = {'reg_bbox': region.bbox, 'center': (
        round((region.bbox[0] + region.bbox[2]) / 2), round((region.bbox[1] + region.bbox[3]) / 2)), 'status': 'r'}
        regioni1.append(regioniXX)
    g_d = nuller()
    nn = len(regioni1)

    if nn > 1:
        if nn == 2:
            if (regioni[0].area > regioni[1].area):
                return regioni[0].bbox[0], regioni[0].bbox[2],regioni[0].bbox[1],regioni[0].bbox[3]
            else:
                return regioni[1].bbox[0],regioni[1].bbox[2],regioni[1].bbox[1],regioni[1].bbox[3]
        else:
            for r in regioni1:
                d=0
                for r1 in regioni1:
                    if(r['center']!=r1['center']):
                        d+=pow((r['center'][0]-r1['center'][0]),2)+pow((r['center'][1]-r1['center'][1]),2) #nalazi rastojanja
                r['dist']=d/(nn-1) #sr vr. rastojanja
                g_d+=d/(nn-1) #sr. vr . rast. nesto...

            a_g_d=1.3*g_d/nn #avr. vr. rast.... na osnovu koje se racuna

            for r in regioni1:
                if r['dist']>a_g_d:
                    r['status']='w'
                else:
                    r['status']='r'

    m1=vratiIspravnu(35)
    m2=vratiIspravnu(35)
    m3=vratiIspravnu(-1)
    m4=vratiIspravnu(-1)

    qwe = (m1+m2+m3+m4)/2

    for r in regioni1:
        if(r['status']=='r'):
            bbox=r['reg_bbox']
            if bbox[0]<m1:
                m1 = vratiIspravnu(bbox[0])
            if bbox[1]<m2:
                m2= vratiIspravnu(bbox[1])
            if bbox[2]> m3:
                m3= vratiIspravnu(bbox[2])
            if bbox[3]> m4:
                m4= vratiIspravnu(bbox[3])

    return m1,m3,m2,m4

def donjiDesniCosak(slika_BW, n, koefx):
    naziv_slike=label(slika_BW)
    regioni = regionprops(naziv_slike)

    novaSlika="novaSlika"
    m1=vratiIspravnu(700)
    m2=vratiIspravnu(700)
    m3=vratiIspravnu(-1)
    m4=vratiIspravnu(-1)

    #cv2.imshow('1. slika', slika_BW)
    #cv2.waitKey(2000)


    #slika_GGx = img_as_ubyte(regioni[0][0])
    #cv2.imshow('slikaStara GG', slika_GGx)
    #cv2.waitKey(2000)
    #cv2.imshow('aslkdajsl', regioni)
    #cv2.waitKey(2000)
    ewq=nuller()
    #m1,m3,m2,m4 = prepoznajRegione(slika_BW);
    if n==1:
        m1,m3,m2,m4 = prepoznajRegione(slika_BW)
        #print("Elemi: " + str(m1)+" " + str(m2)+ " " + str(m3)+ " " + str(m4))
    else:
        for region in regioni:
            #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!USLO U OVO UOPSTE!")
            bbox = region.bbox
            #print("bb0: " + str(bbox[0]))
            #print("bb1: " + str(bbox[1]))
            #print("bb2: " + str(bbox[2]))
            #print("bb3: " + str(bbox[3]))
            if bbox[0]<m1:
                m1=bbox[0]
            if bbox[1]<m2:
                m2=bbox[1]
            if bbox[2]>m3:
                m3=bbox[2]
            if bbox[3]>m4:
                m4=bbox[3]

    ogr = (ewq*2+2*28)*koefx   #f. pom. vis.

    visina = m3-m1
    duzina = m4-m2
    visina = visina +ogr

    novaSlika=np.zeros((28,28))

    #print("VISINA: " + str(visina))
    #print("DUZINA: " + str(duzina))

    #novaSlika[0:visina+1, 0:duzina+1] = novaSlika[0:visina+1, 0:duzina+1] + slika_BW[m1:m3+1, m2:m4+1]
    novaSlika[28-visina:28, 28-duzina:28] = novaSlika[28-visina:28, 28-duzina:28] + slika_BW[m1:m3,m2:m4]
    #novaSlika[28 - visina-1:28, 28 - duzina-1:28] = novaSlika[28 - visina-1:28, 28 - duzina-1:28] + slika_BW[m1-1:m3,m2-1:m4]

    return novaSlika

def nadjiBroj(slikaBroj):
    i=nuller()
    min_sum = 9999999
    r=45
    while i<70000:
        suma=nuller()
        mnist_slika = novi_MNIST_set_podataka[i]
        suma=np.sum(mnist_slika!=slikaBroj)
        if min_sum>suma:
            min_sum=suma
        if suma<16:
            #print("MIN SUM: " + str(min_sum))
            return mnist.target[i], suma
        i=i+1
    r=-1
    #print("MIN SUM: " + str(min_sum))
    if(r == -1):
        i=nuller()
        min_sum=9999999
        poz =-1;
        while i < 70000:
            suma = nuller()
            mnist_slika = novi_MNIST_set_podataka[i]
            suma = np.sum(mnist_slika != slikaBroj)
            if min_sum > suma:
                min_sum = suma
                poz=i
            #if suma < 60:
            #    print("MIN SUM: " + str(min_sum))
            #    return mnist.target[i]
            i = i + 1
        #print("MIN SUM1 : " + str(min_sum))
        if min_sum<120:
            #print("naslo nesto ------------------------------")
            #print("MIN_SUM2 :" + str(min_sum))
            if poz!=-1:
                return mnist.target[poz], min_sum
    return -1, min_sum

def kreirajJezgroRadiObojenihTackica(x, y, p1, p2):

    jel = False
    komsije=[]
    if(x-p1>=0 & y-p1>=0):
        komsije.append((x-p1,y-p1))
        komsije.append((x,y-p1))
        komsije.append((x-p1,y))
        jel = True
    elif(x-p1>=0 & y+p1<28):
        komsije.append((x-p1,y+p1))
        jel = True
    elif(x+p1<28 & y-p1>=0):
        komsije.append((x+p1,y-p1))
        jel = True
    elif(x+p1<28 & y+p1<28):
        komsije.append((x+p1,y+p1))
        komsije.append((x,y+p1))
        komsije.append((x+p1,y))
        jel = True
    elif(x-p2>=0 & y-p2>=0):
        komsije.append((x-p2,y-p2))
        komsije.append((x,y-p2))
        komsije.append((x-p2,y))
        jel = True
    elif(x-p2>=0 & y+p2<28):
        komsije.append((x-p2,y+p2))
        jel = True
    elif(x+p2<28 & y-p2>=0):
        komsije.append((x+p2,y-p2))
        jel = True
    elif(x+p2<28 & y+p2<28):
        komsije.append((x+p2,y+p2))
        komsije.append((x,y+p2))
        komsije.append((x+p2,y))
        jel = True
    return komsije, jel

def odrediBoju(slika_BW,x,y):
    komsije, povJel=kreirajJezgroRadiObojenihTackica(x, y, 1, 2)

    #if povJel == True:
        #print("Jeste")

    c_b=0
    c_w=0
    for x in range(0,len(komsije)):
        if(slika_BW[komsije[x][0],komsije[x][1]]==0):
            c_b+=1
        else:
            c_w+=1
    if(c_w>2):
        slika_BW[x,y]=1.0
        #print("BIJELA BOJA")
        return 'BIJELA'
    else:
        slika_BW[x,y]=0.0
        #print("CRNA BOJA")
        return 'CRNA'

def promjeniBojuPiksela(slika,p1,p2,p3,p4):
    #cv2.imwrite("poslata sl", slika)
    #cv2.waitKey(1250)
    slika1=color.rgb2gray(slika)/255.0
    slika1=(slika1>=0.88).astype('uint8')
    prom = False
    for x in range(0,28):
        for y in range(0,28):
            if slika[x,y,p1]> slika[x,y,p2] & slika[x,y,p3]> slika[x,y,p4]:
                ob=odrediBoju(slika1,x,y)
                if ob == 'CRNA':
                    slika[x,y]=[0,0,0]
                    slika1[x,y]=0.0
                    #print("Uspjelo u CRNU")
                    prom = True
                else:
                    slika1[x,y]=1.0
                    slika[x,y]=[255,255,255]
                    #print("Uspjelo u BIJELU")
                    prom = True
    return slika , prom
from skimage import img_as_ubyte

def showVariantsOfImage(slikilix):
    cv2.imshow('1. stepen',slikilix)
    cv2.waitKey(2500)
    cv2.destroyAllWindows()

    s2xA  = (color.rgb2gray(slikilix) <=0.55)
    print(type(s2xA))
    s2xAo = img_as_ubyte(s2xA)
    cv2.imshow('slika A', s2xAo)
    cv2.waitKey(2000)

    s2xB = (color.rgb2gray(slikilix) > 0.55)
    print(type(s2xB))
    s2xBo = img_as_ubyte(s2xB)
    cv2.imshow('slika B', s2xBo)
    cv2.waitKey(2000)

    s2xC = (color.rgb2gray(slikilix) > 0.75)
    print(type(s2xC))
    s2xCo = img_as_ubyte(s2xC)
    cv2.imshow('slika B', s2xCo)
    cv2.waitKey(2000)

    cv2.destroyAllWindows()




def dajBroj(slika):
    #naziv_slike= label(1-slika)

    #cv2.imshow('1. slika', slika)
    #cv2.waitKey(2000)

    #spremna_slika, pp = promjeniBojuPiksela(slika,1,0,1,2)
    spremna_slika = slika
    miss=99999
    #cv2.imshow('nakon obrade boje', spremna_slika)
    #cv2.waitKey(2000)
    #print(type(spremna_slika))

    yz = (5,5)
    ym = (1,1)

    #if pp==True:
    #    yz=(5,5)
    #    ym = (1, 1)
    #else:
    #    yz=(5,5)
    #    ym = (1, 1)

    #slika_GG = (color.rgb2gray(spremna_slika) <=0.88).astype('uint8')
    #slika_GG = (color.rgb2gray(spremna_slika) <=0.88)
    #print(type(slika_GG))
    #slika_GGx = img_as_ubyte(slika_GG)
    #cv2.imshow('slikaStara GG', slika_GGx)
    #cv2.waitKey(2000)

    slika_BW = (color.rgb2gray(spremna_slika) >=0.88).astype('uint8')
    #slika_BWx = (color.rgb2gray(spremna_slika) >=0.88)
    #slika_BWxx = img_as_ubyte(slika_BWx)
    k = np.ones(yz,np.uint8)
    k1 = np.ones(ym,np.uint8)
    #cv2.imshow('slikaStara', slika_BWxx)
    #cv2.waitKey(2000)
    #cv2.destroyAllWindows()
    novaSlika = donjiDesniCosak(slika_BW, 0, 0)
    #showVariantsOfImage(novaSlika)
    #novaSlika = slika_BWxx
    #cv2.imshow('slikaNova', novaSlika)
    #cv2.waitKey(2000)
    #cv2.destroyAllWindows()
    r,miss=nadjiBroj(novaSlika)
    if r==-1:
        novaSlika=donjiDesniCosak(slika_BW, 1, 0)
        #novaSlika=slika_BW
        r,miss=nadjiBroj(novaSlika)
    return r, miss


def transformisiMNIST(mnist, br):
    x=nuller()
    isp=""
    while x<br:

        if(x==5000):
            print("[*]")
        if(x==15000):
            print("[**]")
        if(x==25000):
            print("[***]")
        if(x==35000):
            print("[****]")
        if(x==45000):
            print("[*****]")
        if(x==55000):
            print("[******]")
        if(x==65000):
            print("[*******]")

        mnist_slika = mnist.data[x].reshape(28,28)
        # mnist_slika_BW = (color.rgb2gray(mnist_slika) > 0.88)
        # mnist_slika_BW = ((color.rgb2gray(mnist_slika) /255.0) > 0.88)
        mnist_slika_BW = ((color.rgb2gray(mnist_slika)/255.0)>0.88).astype('uint8')

        nova_mnist_slika = donjiDesniCosak(mnist_slika_BW, 0, 0)

        #nova_mnist_slika1 = mnist_slika_BW
        #nova_mnist_slika = img_as_ubyte(nova_mnist_slika1)
        #cv2.imshow('slikaNova', nova_mnist_slika)
        #cv2.waitKey(2000)
        novi_MNIST_set_podataka.append(nova_mnist_slika)
        x=x+1

    return True

def obradaVidea(cap,linija,px1,px2,proc,naz,f,poluNazivVidea, outFile):
    radi=False
    pr_br = nuller()
    kkk=np.ones((px1,px2),np.uint8)
    gwq1=[230,230,230]
    gwq2=[255,255,255]
    granice=[(gwq1,gwq2)]

    elementi=[]
    t=nuller()
    brojac=nuller()
    sabirac=nuller()



    while(1):
        ret,slika=cap.read()
        if not ret:
            break
        (donja,gornja)= granice[0]
        donja=np.array(donja,dtype="uint8")
        gornja = np.array(gornja,dtype="uint8")
        maska=cv2.inRange(slika,donja,gornja)



        slika0= proc * maska

        slika0=cv2.dilate(slika0,kkk)
        slika0=cv2.dilate(slika0, kkk)

        oznaceni, najblizi_objekti=ndimage.label(slika0)
        objekti = ndimage.find_objects(oznaceni)

        for x in range(najblizi_objekti):
            lokalni = objekti[x]
            (xc,yc)=((lokalni[1].stop + lokalni[1].start)/2, (lokalni[0].stop + lokalni[0].start)/2)
            (dxc,dyc) = ((lokalni[1].stop - lokalni[1].start),(lokalni[0].stop - lokalni[0].start))

            if(dxc>11 or dyc>11):
                element = {'center':(xc,yc), 'size':(dxc,dyc), 't':t}
                lste = uDosegu(20,element,elementi)
                nn=len(lste)
                if nn==0:
                    element['id']=sledeciID()
                    element['t']=t
                    element['pass']=False
                    element['hist']=[{'center':(xc,yc), 'size':(dxc,dyc),'t':t}]
                    element['fut']=[]
                    elementi.append(element)
                elif nn==1:
                    lste[0]['center']=element['center']
                    lste[0]['t']=t
                    lste[0]['hist'].append({'center':(xc,yc),'size':(dxc,dyc),'t':t})
                    lste[0]['fut']=[]

        for ele in elementi:
            t2 = t-ele['t']
            if(t2<3):
                distanca,tacka,r=tackaDoLinije(ele['center'],linija[0],linija[1])
                if r>0:
                    c=(25,25,255)
                    if(distanca<9):
                        c=(0,255,160)
                        if ele['pass']==False:
                            ele['pass']=True
                            brojac+=1
                            (x,y)=ele['center']
                            (sx,sy)=ele['size']

                            cetrnaest = 14

                            x1=x-cetrnaest
                            x2=x+cetrnaest
                            y1=y-cetrnaest
                            y2=y+cetrnaest

                            br, nepoklapanja=dajBroj(slika[y1:y2,x1:x2])

                            pr_br=br

                            print("--> nadjeni broj: "+str(br) + " , [miss = " + str(nepoklapanja) + "]")
                            f.write("Nadjen br: " +str(br) + " \n")

                            if( br!=None):
                                if(br!=-1):
                                    sabirac+=br

                ide=ele['id']
                for h in ele['hist']:
                    t3=t-h['t']
                    if(t3<100):
                        dsa=1

                for f in ele['fut']:
                    t3=f[0]-t
                    if(t3<100):
                        dsa=2


        cv2.putText(slika, str(naz), (30,30), cv2.FONT_HERSHEY_SIMPLEX,1,(90,90,250),2)
        cv2.putText(slika,'Br. prelaza: '+ str(brojac), (400,450), cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,150),2)
        cv2.putText(slika,'Suma: '+str(sabirac),(400,400),cv2.FONT_HERSHEY_SIMPLEX,1,(90,90,250),2)
        cv2.putText(slika, 'Pr. br: ' + str(pr_br), (400, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 90, 60), 2)
        t +=1

        cv2.imshow('frame',slika)
        k=cv2.waitKey(20)&0xff
        #if k==15:
        #    break
    cap.release()


    print("BROJ PRELAZA: " + str(brojac))
    f.write("BROJ PRELAZA: " + str(brojac) + " \n")
    f.write("SUMA BROJEVA: " + str(sabirac)+ " \n")
    outFile.write(poluNazivVidea + " " + str(sabirac) + "\n")
    print("SUMA: "+ str(sabirac))

    return True

def okidac():

    dalje=True

    print("---------------***---------------")
    print("------POCETAK RADA PROGRAMA------")
    print("---------------***---------------")
    print("")
    print("---------------------------------")
    print(">Trazenje linije: ")
    nazivVidea = "videos/video-0.avi"
    a, b, c, d = nadjiParametreLinije(nazivVidea)
    linija = [(a, b), (c, d)]
    print(">Linija pronadjena!")
    print("---------------------------------")
    print(">Transformacija MNIST podataka: ")
    print("[]")
    mnist_gotov = transformisiMNIST(mnist,70000)

    if mnist_gotov==True:
        print("---------------------------------")
        print(">Transformacija MNIST podataka zavrsena! ")
        print(">Pocetak obrade video materijala: ")
        print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")

        f = open("rezultatObrade.txt", "w")
        outFile = open("outx.txt","w")
        f.write("Rezultati obradjenih video materijala: \n")
        outFile.write("RA 30/2013 Igor Mandic\n")
        outFile.write("file\tsum\n")


        for slVid in range(0,10):
            if dalje==True:
                nazivVidea="videos/video-"+format(slVid)+".avi"
                poluNazivVidea = "video-"+format(slVid)+".avi"
                cap = cv2.VideoCapture(nazivVidea)
                print("------------------------------------------")
                print(">> VIDEO : "+format(slVid))
                f.write("-----------------------------------")
                f.write("Naziv videa: " + str(nazivVidea) + " \n")
                dalje= obradaVidea(cap,linija,2,2,1.0,nazivVidea,f,poluNazivVidea,outFile)

        f.close()
        outFile.close()
        cv2.destroyAllWindows()

okidac();