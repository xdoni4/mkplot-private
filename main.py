from plotting import *

def main():
    data, xl = Parser.read("conf.json")
    plots = Parser.parse_object(data, xl)
    Plotter.plot(plots)

if __name__ == '__main__':
    main()
