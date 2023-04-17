# mkplot
Построение графиков с помощью matplotlib

# Возможности
- Построение графика МНК, полиномиальной регрессии по точкам с указанием погрешностей в виде крестов
- Построение нескольких графиков на одной координатной плоскости
- Добавление названий графиков, осей, легенды

# Как пользоваться
Добавляете в conf.json информацию о графике, который вы хотите построить. Рассмотрим на примере:

    "data" : [
        {
            "title" : "Plot 1",
            "subplots" : [
                {
                    "type" : "lsq",
                    "x" : [0.09, 0.17, 0.26, 0.32, 0.39],
                    "y" : [0.05, 0.09, 0.14, 0.17, 0.21],
                    "xerr" : [0.01, 0.02, 0.02, 0.01, 0.02],
                    "yerr" : [0.03, 0.01, 0.01, 0.04, 0.01],
                    "axes_labels" : ["dx", "F"],
                    "axes_pupils" : ["м", "Н"],
                    "color" : "red",
                    "description" : "spring characteristic"
                },
                {
                    "type" : "lsq",
                    "x" : [0.09, 0.17, 0.26, 0.32, 0.39],
                    "y" : [0.18, 0.34, 0.52, 0.64, 0.78],
                    "xerr" : [0.01, 0.02, 0.02, 0.01, 0.02],
                    "yerr" : [0.03, 0.01, 0.01, 0.04, 0.01],
                    "axes_labels" : ["dx", "F"],
                    "axes_pupils" : ["м", "Н"],
                    "color" : "blue",
                    "description" : "spring characteristic"
                }
            ]
        }
    ]

По структуре это вложенный словарь. Ключу "data" соответствует массив объектов. Каждый из элементов этого массива является координатной плоскостью.
У каждого элемента есть поля "title" -- название и "subplots" -- массив из графиков, которые должны быть построены на этой координатной плоскости.
У каждого графика есть поля:
- "type" : какой график строить по данным. В данный момент поддерживаются: "lsq" -- МНК, "dots" -- просто точки, "poly_n" -- полиномиальная регрессия степени n.
- "x" : массив из x-координат точек
- "y" : массив из y-координат точек
- "xerr" : массив из погрешностей x-координат точек
- "yerr" : массив из погрешностей y-координат точек
- "axes_labels" : массив из названий осей в порядке Ox, Oy
- "axes_pupils" : массив из единиц измерения для осей в порядке Ox, Oy
- "color" : цвет построенной кривой
- "description" : описание кривой в легенде

Когда вы запишете информацию в conf.json, необходимо запустить main.py.
В данном примере будет одна координатная плоскость, на которой будут построены два графика.
Результат будет сохранен в файле /images/fig.png.
Пока есть некоторые траблы с построением нескольких координатых плоскостей -- графики очень сильно сжимаются. 
Поэтому лучше строить только одну координатную плоскость за раз.
