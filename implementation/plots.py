import matplotlib as mpl
from pylab import *
import seaborn as sns
sns.set_context('notebook')
FONT_SIZE = 18

def objective_plot():
    N = 1e3
    x = linspace(-10, 10, num=N)
    y = 4

    phi = {str(lambda_) : (x - y)**2 + lambda_*abs(x) for lambda_ in [0.1, 10]}

    figure()
    for lambda_ in phi:
        plot(x, phi[lambda_], label=r'$\lambda = {}$'.format(lambda_))
    plot(y, 0.1*abs(y), 'og')

    title(r'$\phi(x)$', fontsize=FONT_SIZE)
    text(y, 10, r'$y_i$', fontsize=FONT_SIZE)
    xlabel(r'$x$', fontsize=FONT_SIZE)
    ylabel(r'$({} - x)^2 + |x|$'.format(y), fontsize=FONT_SIZE)
    ylim(-10, 300)
    legend(loc='best')
    savefig('prox_scalar.png', bbox_inches='tight')
    show()

def prox(z, threshold):
    t = abs(z) - threshold
    t[t < 0] = 0
    return sign(z) * t

t = linspace(-1, 1)
x = prox(t, 0.5)

figure()
plot(t, x)
xlabel('z')
ylabel(r'prox(z)')
title(r'Prox operator output, $\lambda = 1$')
savefig('prox.png')
show()
