from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd

# Create your views here.

def predict_GRD(request):
    if request.method == 'POST':
        # Obtener datos del formulario
        datos_usuario = {
            'Edad': int(request.POST.get('edad')),
            'Sexo': request.POST.get('sexo').lower(),
            'Días estancia': int(request.POST.get('dias_estancia', 0)),

            'Dx principal de egreso .1': request.POST.get('dx_principal_1', '').lower(),
            'Dxr 1': request.POST.get('dxr1', '').lower(),
            'Dxr 2': request.POST.get('dxr2', '').lower(),
            'Dxr 3': request.POST.get('dxr3', '').lower(),
            'Dxr 4': request.POST.get('dxr4', '').lower(),
            'Dxr 5': request.POST.get('dxr5', '').lower(),
            'Dxr-6': request.POST.get('dxr6', '').lower(),
            'ServicioAlta': request.POST.get('servicio_alta', '').lower(),
            'Proc1': request.POST.get('proc1', '').lower(),
            'Tipo de ingreso': request.POST.get('tipo_ingreso', '').lower()
        }

        df_usuario = pd.DataFrame([datos_usuario])
        columnas_categoricas = [
            'Sexo', 'Dx principal de egreso .1', 'Dxr 1', 'Dxr 2', 'Dxr 3',
            'Dxr 4', 'Dxr 5', 'Dxr-6', 'ServicioAlta', 'Proc1', 'Tipo de ingreso'
        ]
        for col in columnas_categoricas:
            df_usuario[col] = df_usuario[col].astype(str).str.lower()

        df_usuario = pd.get_dummies(df_usuario)

        columnas_modelo = joblib.load('columnas_modelo.pkl')
        for col in columnas_modelo:
            if col not in df_usuario.columns:
                df_usuario[col] = 0

        df_usuario = df_usuario[columnas_modelo]

        modelo = joblib.load('modelo_random_forest.pkl')
        resultado = modelo.predict(df_usuario)[0]

        # Buscar la descripción del código GRD
        df_descripciones = pd.read_csv('descripcion_grd.csv')  # Asegúrate de que el archivo esté en la misma carpeta
        descripcion = df_descripciones.loc[df_descripciones['GRD -Código'] == resultado, 'GRD -Descripción']
        descripcion = descripcion.values[0] if not descripcion.empty else 'Descripción no encontrada'

        return render(request, 'results.html', {
            'resultado': resultado,
            'descripcion': descripcion
        })

    return render(request, 'form.html')