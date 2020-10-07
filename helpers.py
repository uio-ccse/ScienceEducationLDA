#Use this module as follows:
#from helpers import path_pdf,path_pkl,find_in_list
#from helpers import plot_freq_dist,get_top_n_words,plot_words_freq
#from helpers import printh,get_best_match,find_start,find_next,
#from helpers import elbow_plot, gridsearch_plot, plot_single_alpha
#from helpers import aggregate_topics,cos_sim,gmm_show_topic

import numpy as np
import re
import os
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

## Directories
if os.getenv('HOME').split('/')[2] == 'Tor':
    path_pkl = os.getenv('HOME')+'/Dropbox/CCSE/Machine Learning Project/tor_ale_shared/'
    path_pdf = os.getenv('HOME')+'/Universitetet i Oslo/Alessandro Marin - DOIs_Renamed2/'
elif os.getenv('HOME').split('/')[2] == 'amarin':
    path_pkl = os.getenv('HOME')+'/Dropbox/tor_ale_shared/'
    path_pdf = '/home/amarin/Desktop/papers/DOIs_Renamed2/'
else:
    raise NotADirectoryError('Please define an existing directory for path_pkl')

## Utility to find the index of a single word in a list
find_in_list = lambda l, e: l.index(e) if e in l else -1

def plot_freq_dist(freq_list, **kwargs):
    '''
    Plot distribution of word frequencies
    '''
    fig = plt.subplots(figsize=(15,5))
    _ = plt.hist(freq_list, bins=100, **kwargs);
    _ = plt.title("Distribution of word frequencies ("+str(len(freq_list))+" words)");
    _ = plt.xlabel("Word frequency in corpus", {'fontsize': 14});
    _ = plt.ylabel("Log count", {'fontsize': 14});
    plt.yscale('log', nonposy='clip');
    plt.show();
    return fig

def get_top_n_words(corpus, n_top_words=None):
    '''
    Plot frequency distribution of top n word 
    corpus: list of tokens
    n_top_words: number of most frequent words to plot
    '''
    count_vectorizer = CountVectorizer(stop_words='english')
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx], idx) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return zip(*words_freq[:n_top_words])

def plot_words_freq(word_list, freq_list, n_top_words=20, ylim=None, plot_doc_fraction=False,data_words_bigrams=None,title=True):
    '''
    Plot the frequency of n_top_words in the corpus data, see figure 4 in Odden, Marin, Caballero 2020. 

    Parameters
    ----------
    :param list(str) word_list: list of words in the corpus
    :param list(int) freq_list: frequency of list of words in the corpus
    :param int n_top_words: number of top words to plot
    :param tuple(float) ylim: the limit on the y-axis
    :param bool plot_doc_fraction: Plot the Document fraction on the right axis, default is False
    :param list(list(str)) data_words_bigrams: bigrams for each document, needed if plot_doc_fraction=True
    :return: Tuple (fig, ax) with matplotlib handles to figure and axis
    :rtype: : Tuple[bytes, bytes]
    '''
    fig, ax = plt.subplots(figsize=(8,5))
    word_len = str(len(word_list))
    freq_list = freq_list[:n_top_words]
    word_list = word_list[:n_top_words]
    ax.plot(range(len(freq_list)), freq_list, label='Number of occurrences');
    ax.set_xticks(range(len(word_list)));
    xticks = list(map(lambda w: str(w), word_list));
    ax.set_xticklabels(xticks, rotation=45, ha='right', fontdict={'fontweight': 'normal'});
    if title == True:
        ax.set_title('Top words in corpus (' + word_len + ' total words)', {'fontsize': 16, 'fontweight': 'bold'});
    ax.set_xlabel('Top words', {'fontsize': 14});
    ax.set_ylabel('Number of occurrences (log scale)', {'fontsize': 14});
    ax.set_yscale('log', nonposy='clip', );
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([10**(np.floor(np.log10(min(freq_list[:20]))*10)/10), 10**(np.ceil(np.log10(max(freq_list))*10)/10)]);    
    #Fraction
    if plot_doc_fraction:
        frac = [sum([w in temp for temp in data_words_bigrams])/len(data_words_bigrams) for w in word_list[:n_top_words]]
        ax2 = ax.twinx()
        ax2.plot(range(len(freq_list)), frac, 'r', label='Document fraction');
        ax2.set_ylim([np.floor(min(frac)*10)/10,1])
        ax2.set_ylabel('Document fraction', {'fontsize': 14}, labelpad=10);
        plt.legend(loc='upper left')    
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')    
    plt.tight_layout()
    plt.show();
    return fig, ax


## Fuzzy string match
from difflib import SequenceMatcher
def get_best_match(query, corpus, step=4, flex=3, case_sensitive=False, verbose=False):
    """Return best matching substring of corpus.
    Credits to the accepted answer here: https://stackoverflow.com/questions/36013295/find-best-substring-match

    Parameters
    ----------
    :param str query:
    :param str corpus:
    :param int step: Step size of first match-value scan through corpus. Can be thought of
        as a sort of "scan resolution". Should not exceed length of query.
    :param int flex: Max. left/right substring position adjustment value. Should not
        exceed length of query / 2.
    :return output0: Best matching substring.
    :rtype: str
    :return output1: Match ratio of best matching substring. 1 is perfect match.
    :rtype: float
    """
    def _match(a, b):
        """Compact alias for SequenceMatcher."""
        return SequenceMatcher(None, a, b).ratio()

    def scan_corpus(step):
        """Return list of match values from corpus-wide scan."""
        match_values = []

        m = 0
        while m + qlen - step <= len(corpus):
            match_values.append(_match(query, corpus[m : m-1+qlen]))
            if verbose:
                print(query, "-", corpus[m: m + qlen], _match(query, corpus[m: m + qlen]))
            m += step

        return match_values

    def index_max(v):
        """Return index of max value."""
        return max(range(len(v)), key=v.__getitem__)
    
    def index_maxima(v, n=5):
        """Return indices of n max values."""
        #change index_max() to return a list of indices for the n highest values of the input list, 
        #and loop over adjust_left_right_positions() for values in that list.
        max_ind = np.argpartition(v, -n)[-n:] #unsorted indices of n maxima (n=4)
        
        max_ind = list(np.argpartition(v, -n)[-n:]) #unsorted indices of n maxima (n=4)
        return [max_ind[i] for i in np.argsort(np.array(np.array(v))[max_ind])]
        
    
    def adjust_left_right_positions():
        """Return left/right positions for best string match."""
        # bp_* is synonym for 'Best Position Left/Right' and are adjusted 
        # to optimize bmv_*
        #bp_ls, bp_rs, matches = [] ,[], []
        #for pos in positions:
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + qlen] * 2

        # bmv_* are declared here in case they are untouched in optimization
        r = int(p_l / step)
        if int(r) != r: print(ratio is not integer.investigate)
        bmv_l = match_values[r]
        bmv_r = match_values[r]

        for f in range(flex):
            ll = _match(query, corpus[p_l - f: p_r])
            if ll > bmv_l:
                bmv_l = ll
                bp_l = p_l - f

            lr = _match(query, corpus[p_l + f: p_r])
            if lr > bmv_l:
                bmv_l = lr
                bp_l = p_l + f

            rl = _match(query, corpus[p_l: p_r - f])
            if rl > bmv_r:
                bmv_r = rl
                bp_r = p_r - f

            rr = _match(query, corpus[p_l: p_r + f])
            if rr > bmv_r:
                bmv_r = rr
                bp_r = p_r + f

            if verbose:
                print("\n" + str(f))
                print("ll: -- value: %f -- snippet: %s" % (ll, corpus[p_l - f: p_r]))
                print("lr: -- value: %f -- snippet: %s" % (lr, corpus[p_l + f: p_r]))
                print("rl: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r - f]))
                print("rr: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r + f]))
            #bp_ls.append(bp_l)
            #bp_rs.append(bp_r)
            #matches.append(_match(query, corpus[bp_l : bp_r]))
        return bp_l, bp_r, _match(query, corpus[bp_l : bp_r])
        #return bp_ls, bp_rs, matches
        
    if not case_sensitive:
        query = query.lower()
        corpus = corpus.lower()

    qlen = len(query)

    if flex >= qlen/2:
        print("Warning: flex %d exceeds length of query / 2 = %d. Setting to default. query=%s" % 
              (flex, qlen/2, query))
        flex = 3

    match_values = scan_corpus(step)
    pos = index_max(match_values) * step
    #positions = list(map(lambda x: x * step, index_maxima(match_values)))
    pos_left, pos_right, match_value = adjust_left_right_positions()
    return (pos_left,pos_right), corpus[pos_left: pos_right].strip(), match_value


#Print with highlighted regex
#Print with highlighted regex
def printh(text, pattern='', crop = -1, print_pos = False):
    '''Print text, highlight a pattern, crop around all found patterns.
    
    :param str text:
    :param str pattern: regex pattern
    :param str crop: int for trimming text around patterns found, default is -1
    :param str print_pos: boolean for printing positions in text of found patterns, default is False
    '''
    ms = [m for m in re.finditer(pattern, text)]
    highlight_start = '\x1b[1;31;43m'
    highlight_end = '\x1b[0m'
    if print_pos:
        print([m.start() for m in ms])
    if len(ms) is 0:
        print('\033[1mprinth: pattern \''+pattern+'\' not found\x1b[0m')
        return
    if crop > -1:
        for m in ms:
            cropped_start = max(0, m.start() - crop)
            cropped_end = min(len(text), m.end() + crop)
            temp = text[cropped_start:m.start()] + highlight_start + text[m.start():m.end()] + highlight_end + text[m.end():cropped_end]
            print(temp,'\n')
    else :
        for m in reversed(ms):
            text = text[:m.start()] + highlight_start+ text[m.start():m.end()] + highlight_end + text[m.end():]    
        print(text,'\n')



def find_start(DF, index, text_col = 'raw', threshold=0.67, verbose=False, dist_auth_title=100, apply_lower=True, chars_overlap=100):
    """
    Find title or author for the document at <index>,<text_col> in a DataFrame <DF> by finding an exact match and return the first match. 
    If no exact match is found, perform Fuzzy substring match and return the best (NB: not the first!) match. 
    
    :param DataFrame DF: pandas DataFrame - DataFrame containing the columns: title, authors
    :param int index: Index of DF
    :param str text_col: Column name containing text
    :param float threshold: threshold score for accepting fuzzy substring match, default is 0.67
    :param bool verbose: default is False
    :param str dist_auth_title: Maximum number of chars allowed between authors and title found in text
    apply_lower : convert text and patterns to lowercase, default is 100
    :return startloc: Position of text match. 0 if no match
    :rtype: int
    """
    #Search in text overlap between previous and current articles
    searchtext = DF.loc[index,text_col]
    pos = DF.loc[index-1,text_col].find(DF.loc[index,text_col][0:chars_overlap])
    if pos > 0:
        searchtext = searchtext[0:len(DF.loc[index-1,text_col][max(pos,0):])]
    if verbose: print('%d - searchtext is [0:%d], text column is %d long'%(index,
        len(searchtext),
        len(DF.loc[index,text_col])))
    #overlap_prev=DF.loc[index-1,text_col]
    #overlap=DF.loc[index,text_col]
    #sm = SequenceMatcher(None, overlap_prev, overlap)
    #match=sm.find_longest_match(len(overlap_prev)-4000, len(overlap_prev), 0, 6000) #1 page is typically <=4000 chars
    #searchtext=overlap[match.b:match.b+match.size]
    #if verbose: print('%d - %d char long text overlap with previous article'%(index,len(searchtext)))
    #if len(searchtext)<200:
    #    searchtext = DF.loc[index, text_col][0:6000]
    #    if verbose: print('%d - \033[1mBad overlap with previous article found\033[0m'%index)        
    titlefind = DF.loc[index]['title']
    authorfind = DF.loc[index]['authors'].split(' ')[0]
    if apply_lower: searchtext,titlefind,authorfind=[el.lower() for el in [searchtext,titlefind,authorfind]]
    authorfind+='\W'
    startloc = 0
    #Use regex to find author
    authorloc,titleloc=(-1,-1),(-1,-1)
    ms = [m for m in re.finditer(authorfind, searchtext)]
    if len(ms)==0:
        if verbose: 
            print('%d - Quit. %d matches found on authors %s'%(index,len(ms),authorfind), [(m.start(),m.end()) for m in ms])
        return startloc
    elif len(ms)>1:
        if verbose: print('%d - %d matches found on authors %s. Take the first one. '%(index,len(ms),authorfind), [(m.start(),m.end()) for m in ms])
        ms = [ms[0]]
    if len(ms) == 1: 
        authorloc=(ms[0].start(),ms[0].end())
        if verbose: print('%d - Authors found at %d-%d. index=%d - %s' % (index, authorloc[0], authorloc[1], index, DF.loc[index,'filename']))
    #Use regex or fuzzy find to find title 
    ms = [m for m in re.finditer(titlefind, searchtext)]
    if len(ms)>1:
        if verbose: 
            print('%d - %d matches found on title'%(index,len(ms)))
            print([(m.start(),m.end()) for m in ms])
    if len(ms) == 1: 
        titleloc=(ms[0].start(),ms[0].end())
        if verbose: print('%d - Titles found at %d-%d. index=%d - %s' % (index, titleloc[0], titleloc[1], index, DF.loc[index,'filename']))
    else:
        #NB: because authorfind can be short, applying get_best_match would usually show a warning
        if verbose: print('%d - Fuzzy substring search for current title'%index)
        bestmatch = get_best_match(titlefind, searchtext)
        if bestmatch[2] > threshold:
            titleloc = bestmatch[0]
            if verbose: print('  position fuzzy match=', bestmatch[0])
        else:
            verbose = True
        if verbose:
            print('  Low matching '*(bestmatch[2] <= threshold) + '  score', bestmatch[2])
            print('  %d - %s' % (index, DF.loc[index,'filename']))
            print('  bestmatch current title:',bestmatch[1])
            print('  titlefind:', titlefind)
    #Check that title and authors locations are close.
    if authorloc[0] > -1 and titleloc[0] > -1:
        dist = abs(min(titleloc[1]-authorloc[0],authorloc[1]-titleloc[0]))
        if verbose: print('%d - Distance between title and authors found is %d'%(index,dist))
        if dist < dist_auth_title:
            startloc = min(authorloc[0],titleloc[0])
            if verbose: print('%d - Successfully detected start location: %d - %s' % (index, startloc, DF.loc[index,'filename']))
    return startloc


def find_next(DF, index, text_col = 'raw', threshold=0.67, verbose=False, dist_auth_title=100, apply_lower=True, chars_from_end=5000):
    """
    Find the following article's title or author for the document at <index>,<text_col> in a DataFrame <DF> by right-finding an exact match and (NB!) return the first match. 
    If no match is found, perform Fuzzy substring match and return the best (NB: not the first!) match with score > threshold. 
    
    :param DataFrame DF: DataFrame containing the columns: title, authors
    :param int index: Index of DF
    :param str text_col: Column name containing text
    :param float threshold: threshold score for accepting fuzzy substring match, default is 0.67
    :param bool verbose: default is False
    :param int dist_auth_title:  Maximum number of chars allowed between authors and title found in text
    :param int apply_lower: convert text and patterns to lowercase, default is 100
    :param int chars_from_end: Search text in DF.loc[index, text_col][-chars_from_end:], default is 5000
    :return: startloc: Position of text match. 0 if no match    
    :rtype: int
    """
    searchtext = DF.loc[index, text_col][-chars_from_end:] #1 page is typically <=4000 chars
    nexttitlefind = DF.loc[index+1]['title']
    nextauthorfind = DF.loc[index+1]['authors'].split(' ')[0]
    if apply_lower: searchtext,nexttitlefind,nextauthorfind=[el.lower() for el in [searchtext,nexttitlefind,nextauthorfind]]
    nextauthorfind+='\W'
    endloc = -1
    #Use regex to find next authors
    nextauthorloc,nexttitleloc=(-1,-1),(-1,-1)
    ms = [m for m in re.finditer(nextauthorfind, searchtext)]
    if len(ms) > 1: 
        if verbose: 
            print('%d - %d matches found on the next author. Choose the last.'%(index,len(ms)), 
                [(max(0, len(DF.loc[index, text_col])-chars_from_end)+m.start(), 
                max(0, len(DF.loc[index, text_col])-chars_from_end)+ m.end()) for m in ms])
        ms=[ms[-1]]
    if len(ms) == 1: 
        nextauthorloc=(ms[0].start(),ms[0].end())
        if verbose: print('%d - Authors found at %d-%d - %s' % (index, nextauthorloc[0], nextauthorloc[1], DF.loc[index,'filename']))
    else:
        if verbose: 
            print('%d - Quit. %d matches found on authors %s'%(index,len(ms),nextauthorfind), [(m.start(),m.end()) for m in ms])
        return endloc
    #Use regex or fuzzy to find next title    
    ms = [m for m in re.finditer(nexttitlefind, searchtext)]
    if len(ms)>1:
        if verbose: 
            print('%d matches found on the next title. Choose the last.'%len(ms),
                [(max(0, len(DF.loc[index, text_col])-chars_from_end)+m.start(),
                max(0, len(DF.loc[index, text_col])-chars_from_end)+m.end()) for m in ms])
        ms=[ms[-1]]
    if len(ms) == 1: 
        nexttitleloc=(ms[0].start(),ms[0].end())
        if verbose: print('%d - Titles found at %d-%d - %s' % (index, nexttitleloc[0], nexttitleloc[1], DF.loc[index,'filename']))
    else:
        #Use Fuzzy substring match
        #NB: because authorfind can be short, applying get_best_match would usually show a warning
        if verbose: print('%d - Fuzzy substring search for next title'%index)
        bestmatch = get_best_match(nexttitlefind, searchtext)
        if bestmatch[2] > threshold:
            nexttitleloc = bestmatch[0]
            if verbose: print('  position fuzzy match=', bestmatch[0])
        else:
            verbose = True
        if verbose:
            print('  Low matching '*(bestmatch[2] <= threshold) + '  score', bestmatch[2])
            print('  %d - %s' % (index, DF.loc[index,'filename']))
            print('  bestmatch:',re.sub('[\t\n\r\f\v\d\uf0b7]', ' ', bestmatch[1]))
            print('  titlefind:', nexttitlefind)
    #Check that title and authors locations are close.
    if nextauthorloc[0] > -1 and nexttitleloc[0] > -1:
        dist = abs(min(nexttitleloc[1]-nextauthorloc[0],nextauthorloc[1]-nexttitleloc[0]))
        if verbose: print('%d - Distance between title and authors found is %d'%(index,dist))
        if dist > dist_auth_title:
            return -1
        endloc = min(nextauthorloc[0],nexttitleloc[0])
        if verbose: print('%d - Successfully detected end location: %d - %s' % (index, endloc, DF.loc[index,'filename']))
    return endloc + max(0, len(DF.loc[index, text_col])-chars_from_end)
    #print("NB: filter_start and filter_end return the best fuzzy match, which is not necessarily the first or the last match.")


def elbow_plot(df0, ylim=None):
        ks = df0.num_topics.unique().tolist()
        reps = len(df0[(df0.num_topics == df0.iloc[0].num_topics)])
        #Print scatter plot
        for i,k in enumerate(ks):
            label=(i==0) and 'Coherence' or '_nolabel_'
            plt.scatter([k]*reps, df0[df0.num_topics == k].coherence, c="black", label=label);
        plt.errorbar(ks, df0.groupby('num_topics').coherence.mean(), yerr=df0.groupby('num_topics').coherence.std(), fmt='--o', label="Mean");
        plt.ylabel("Coherence score");
        plt.xlabel("num_topics");
        plt.legend(loc="best")
        plt.ylim(ylim);
        
def gridsearch_plot(df, no_below = None, aggreg_func = 'mean'):
    '''
    Plot the results of a grid search. For each no_below value and each alpha value, plot all coherences by topic number
    aggreg_func - 'mean' or 'median'. Aggregation function for the coherence values on the y-axis
    '''    
    nbs = [None]
    alphas = [None]
    if no_below: 
        df = df[df.no_below == no_below]
    if 'no_below' in df.columns:
        reps = int(len(df)/len(df.no_above.unique())/len(df.no_below.unique())/len(df.num_topics.unique()))
        #reps = len(df[(df.num_topics == df.iloc[0].num_topics) & (df.no_below == df.iloc[0].no_below)])
        nbs = df.no_below.unique()
    else: 
        reps = len(df[(df.num_topics == df.iloc[0].num_topics)])
    if 'alpha' in df.columns:
        reps = int(reps / len(df.alpha.unique()))
        alphas = df.alpha.unique()        
    for nb in nbs:
        from matplotlib.ticker import MaxNLocator
        if nb: df0 = df[df.no_below == nb]
        for alpha in alphas:
            if alpha: df0 = df[df.alpha == alpha]
            x = df0.num_topics.unique()
            y = df0.coherence
            fig, ax1 = plt.subplots()
            for i in range(reps):
                _ = ax1.scatter(x, y[i::reps], c="black");
            if aggreg_func == 'mean':
            	coh_mean = y.groupby(np.arange(len(y))//reps).mean()
            elif aggreg_func == 'median':
            	coh_mean = y.groupby(np.arange(len(y))//reps).median()
            print('Maximum %f at num_topics=%d ' % (max(coh_mean), x[0]+list(coh_mean).index(max(coh_mean))))
            _ = ax1.plot(x, coh_mean, c="blue", label="mean");
            _ = ax1.errorbar(x, coh_mean, yerr=y.groupby(np.arange(len(y))//reps).std(), fmt='--o')
            _ = ax1.set_ylabel("Coherence score");
            _ = ax1.set_xlabel("num_topics");
            _ = ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
            _ = plt.legend(loc="best")
            _ = plt.title("Coherence score. no_below=%s" % (str(nb)))
            if alpha: _ = plt.title("Coherence score. no_below=%s, alpha=%s" % (str(nb), str(alpha)))
            plt.show();

def plot_single_alpha(df0, ax, alpha):
    df0 = df0[df0.alpha == alpha]
    ks = df0.num_topics.unique().tolist()
    reps = len(df0[(df0.num_topics == df0.iloc[0].num_topics)])
    #Print scatter plot for each alpha
    for i,k in enumerate(ks):        
        label=(i==0) and 'Coherence' or '_nolabel_'
        ax.scatter([k]*reps, df0[df0.num_topics == k].coherence, s=60, facecolors='none', edgecolors='k', label=label);
    ax.errorbar(ks, df0.groupby('num_topics').coherence.mean(), yerr=df0.groupby('num_topics').coherence.std(), fmt='--o', label="Mean");
    _ = ax.set_ylabel("Coherence score", {'fontsize': 14});
    _ = ax.set_xlabel("Number of topics", {'fontsize': 14});
    _ = ax.legend(loc="best")
    _ = ax.set_title(r"$\alpha=%s$" % (str(alpha)), {'fontsize': 14})
    _ = plt.ylim([0.39, 0.53])


def aggregate_topics(matrix_topics, lbls, n_components = None):
    '''Aggregate (i.e. average) topics based on labels. For example, pass a cluster model's labels and this 
     function will return a (K', n_words) matrix where each column is the average of all points a clusters'''
    if not n_components:
        n_components = max(lbls)+1
    else:
        if n_components != max(lbls)+1:
            raise ValueError("%d labels inconsistent with n_components (%d)" % (max(lbls)+1,n_components))
    return [np.mean(matrix_topics[lbls == l], axis=0) for l in range(n_components)]

from numpy.linalg import norm #import linear algebra norm
cos_sim = lambda v1,v2: np.inner(v1, v2) / (norm(v1) * norm(v2)) #define cosine similarity

def gmm_show_topic(model_cluster, topicid, hypertopics, topn = 20):
    """Adapting LdaModel.show_topic to a clustering model"""
    idx = np.argsort(hypertopics[topicid])[::-1][:topn] #Find the cluster centers for each cluster, and locate the ids for their words
    values = hypertopics[topicid][idx] 
    topic = [(str(id2word[i])+'*'+'{:5.3}'.format(hypertopics[topicid][i])) for i in idx]  #build the topic list from those ids      
    topic = [(id2word[i],hypertopics[topicid][i]) for i in idx]  #build the topic list from those ids      
    return topic