import matplotlib.pyplot as plt

i = -3
rew_list = []
eval_eps = 1
t_actor_loss_list = []
t_critic_loss_list = []
d_actor_loss_list = []
d_critic_loss_list = []

log_file = open('./train_log/train/log.log')

for line in log_file:
    '''
    i+=1
    if i ==100:
        data = line.split(' ')
        #print(data,"\n")
        print(data[4],"         ")
        #print(data[7])
        if data[4] == 'Evaluation':
            print(eval_eps , "\n")
            a = data[9]
            a = a[:len(a) -1]
            rew = float(a)
            rew_list.append(rew)
            i = -1
            eval_eps += 1 
        else:
            eps = data[7]
            eps = eps[:len(eps)-1]
            eps = int(eps)
            i = eps % 100
            #print(i,"\n")
    '''
    data = line.split(' ')
    if data[4] == 't_critic_loss' :
        t_critic_loss = data[5]
        t_critic_loss = t_critic_loss[:len(t_critic_loss) -1]
        t_critic_loss = float(t_critic_loss)
        t_critic_loss_list.append(t_critic_loss)

        t_actor_loss = data[7]
        t_actor_loss = t_actor_loss[:len(t_actor_loss) -1]
        t_actor_loss = float(t_actor_loss)
        t_actor_loss_list.append(-t_actor_loss)

        d_critic_loss = data[9]
        d_critic_loss = d_critic_loss[:len(d_critic_loss) -1]
        d_critic_loss = float(d_critic_loss)
        d_critic_loss_list.append(d_critic_loss)

        d_actor_loss = data[11]
        d_actor_loss = d_actor_loss[:len(d_actor_loss) -1]
        d_actor_loss = float(d_actor_loss)
        d_actor_loss_list.append(-d_actor_loss)

    if data[4] == 'Evaluation':
        print(eval_eps , "\n")
        a = data[9]
        a = a[:len(a) -1]
        rew = float(a)
        rew_list.append(rew)
        i = -1
        eval_eps += 1 

        #print(critic_loss ,  actor_loss ,"\n")


#plt.plot(rew_list)
#plt.show()
plt.figure(1)
plt.plot(t_actor_loss_list)
plt.plot(t_critic_loss_list)
plt.legend(['t_actor_loss' , 't_critic_loss'])

plt.figure(2)
plt.plot(d_actor_loss_list)
plt.plot(d_critic_loss_list)
plt.legend(['d_actor_loss' , 'd_critic_loss'])

plt.figure(3)
plt.plot(rew_list)
plt.show()